"""
02_pipeline.py  (v4 — parallel windows + parallel QA + checkpoint resume)
==========================================================================

Parallelism strategy:
  - ThreadPoolExecutor for date groups within each batch (SPECTRAL_WORKERS=6)
    Each (start_year, completion_year) group is independent — different
    composite pairs, different reduceRegions calls. Threads fire multiple
    GEE HTTP requests simultaneously. GIL releases during HTTP I/O so
    threading actually works here (unlike multiprocessing which breaks GEE).

  - ThreadPoolExecutor for the 3 spatial windows inside each date group
    Previously sequential; now all 3 reduceRegions calls fire in parallel.
    This is the dominant cost per group — parallelising saves ~2/3 of
    window time, the single biggest speedup in this version.

  - ThreadPoolExecutor for QA checks (before/after scene count + pixel check)
    Previously 4 serial GEE calls; now before+after run simultaneously.

  - ThreadPoolExecutor for thumbnails
    Each getThumbURL is one HTTP request. Highly parallelizable.
    Default 5 workers — safe under GEE non-commercial quota.

  - Batch size 50: projects sharing same date pair share ONE composite build.
    Larger batches = more date-pair sharing = fewer composite builds total.
    50 is the sweet spot before GEE memory pressure increases.

  - Checkpoint: results written after every batch so a crash never loses more
    than one batch. On restart the script detects the checkpoint and skips
    already-processed project IDs automatically.

Why NOT multiprocessing:
  GEE objects (ee.Image, ee.FeatureCollection etc.) are not picklable.
  multiprocessing.Pool fails on Windows. ThreadPoolExecutor is correct here.

Max concurrent GEE requests per date-group thread:
  QA phase:      2  (before + after scene/pixel checks in parallel)
  Window phase:  3  (tight + broad + anywhere reduceRegions in parallel)
  Thumb phase:   THUMB_WORKERS (default 5)
  Across all spectral workers: SPECTRAL_WORKERS × max(3, THUMB_WORKERS)
  Default: 6 × 5 = 30 — well within GEE non-commercial quota.

Usage:
  python 02_pipeline.py --input flood_projects.csv --output results.csv --project ee-fragotyron
  python 02_pipeline.py --input flood_projects.csv --output results.csv --project ee-fragotyron --test
  python 02_pipeline.py --input flood_projects.csv --output results.csv --project ee-fragotyron --skip-thumbs
  python 02_pipeline.py --input flood_projects.csv --output results.csv --project ee-fragotyron --spectral-workers 8
"""

import ee
import os
import pandas as pd
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── CONFIG ────────────────────────────────────────────────────────────────────
DRY_SEASON_START   = "11-01"
DRY_SEASON_END     = "04-30"
CLOUD_SCENE_FILTER = 60
MIN_SCENES         = 5
S2_MIN_YEAR        = 2017

# Batch size: how many projects loaded into Python at once.
# Within each batch, date groups share composites — larger batch = more sharing.
# 50 is optimal: enough date-pair clustering without GEE memory pressure.
BATCH_SIZE = 50

# Thread workers:
# - SPECTRAL_WORKERS: parallel date groups (each fires a reduceRegions to GEE)
# - THUMB_WORKERS:    parallel thumbnail requests
# GEE non-commercial quota is generous for concurrent requests.
# Don't exceed 8 — GEE will start throttling and raising EEException.
SPECTRAL_WORKERS = 6
THUMB_WORKERS    = 5

THUMB_BUFFER_M = 500   # buffer around project point for tile viewport

WINDOWS = [
    {"name": "tight",    "radius": 30,  "reducer": "mean"},
    {"name": "broad",    "radius": 100, "reducer": "mean"},
    {"name": "anywhere", "radius": 150, "reducer": "max"},
]

INDEX_NAMES = ["IBI", "BUI", "NDVI", "NDBI", "MNDWI", "TCB", "TCG", "TCW", "NDTI"]

TC = {
    "TCB": [ 0.3510,  0.3813,  0.3437,  0.7196,  0.2396,  0.1949],
    "TCG": [-0.3599, -0.3533, -0.4734,  0.6633,  0.0087, -0.2856],
    "TCW": [ 0.2578,  0.2305,  0.0883,  0.1071, -0.7611, -0.5308],
}

PATTERN_MEANINGS = {
    "T+B+A+": "Built at coordinate -- construction confirmed",
    "T-B+A+": "Built nearby -- moderate coordinate offset",
    "T-B-A+": "Built within 150m but coordinate displaced (verify GPS first)",
    "T+B-A-": "Small point structure or noise at coordinate",
    "T+B+A-": "Check composite quality",
    "T-B-A-": "No signal -- likely ghost project or coordinate error > 150m",
    "T+B-A+": "Scattered signal -- possible partial construction",
    "T-B+A-": "Ambiguous -- check coordinate accuracy",
}


# ── GEE INIT ─────────────────────────────────────────────────────────────────
# ── STRATIFIED SAMPLING ───────────────────────────────────────────────────────
def stratified_sample(df, n=500, seed=42):
    """
    Proportional stratified sample by completion_year x region.

    Why stratified instead of df.head(n):
      head(n) takes projects from one or two years only — skews the results
      toward that period and misses regional variation.
      Stratification ensures every year/region combination is represented
      proportionally, making the sample statistically valid for the full dataset.

    n=500 gives 95% confidence interval with ±4.3% margin of error
    for a population of 24,000 (standard formula: n = z^2*p*(1-p)/e^2).
    """
    df = df.copy()
    df["stratum"] = (
        df["completion_year"].astype(str) + "_" +
        df["region"].fillna("unknown").astype(str)
    )

    stratum_counts = df["stratum"].value_counts()
    # Proportional allocation — each stratum gets seats proportional to its size
    stratum_n = (stratum_counts / len(df) * n).round().astype(int)
    stratum_n = stratum_n.clip(lower=1)   # at least 1 from every stratum

    sampled = []
    for stratum, group in df.groupby("stratum"):
        k = min(stratum_n.get(stratum, 1), len(group))
        sampled.append(group.sample(n=k, random_state=seed))

    result = pd.concat(sampled).drop_duplicates("project_id")

    # Top up if rounding left us short of n
    if len(result) < n:
        remaining = df[~df["project_id"].isin(result["project_id"])]
        if len(remaining) > 0:
            top_up = remaining.sample(
                n=min(n - len(result), len(remaining)),
                random_state=seed
            )
            result = pd.concat([result, top_up])

    result = result.head(n).reset_index(drop=True)
    result = result.drop(columns=["stratum"], errors="ignore")
    return result


def init_gee(project=None):
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        print("[GEE] Authenticated and ready")
    except Exception as e:
        print(f"[GEE] Failed: {e}")
        raise


# ── DATE WINDOWS ─────────────────────────────────────────────────────────────
def before_window(sy):
    """Dry season BEFORE construction. Same seasonal window prevents spectral drift."""
    return f"{sy-1}-{DRY_SEASON_START}", f"{sy}-{DRY_SEASON_END}"

def after_window(cy):
    """Dry season AFTER completion."""
    return f"{cy}-{DRY_SEASON_START}", f"{cy+1}-{DRY_SEASON_END}"


# ── CLOUD MASK ───────────────────────────────────────────────────────────────
def mask_clouds(img):
    """SCL pixel masking. Scales DN -> [0,1]."""
    scl  = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).divide(10000).copyProperties(img, ["system:time_start"])


# ── COMPOSITE ────────────────────────────────────────────────────────────────
def build_composite(aoi, s, e):
    """
    Cloud-free Sentinel-2 SR median composite.

    Two-stage: try dry season window first; fall back to full start-year if
    fewer than MIN_SCENES pass the cloud filter.

    Fallback uses s[:4] only (start year) — avoids spanning 2 calendar years
    which would mix wet and dry season spectral signatures.

    If BOTH collections are empty (sparse 2018 coverage), median() returns
    an image with no bands. The empty-image guard returns zeros instead so
    downstream index computation doesn't crash with 'Band B12 not found'.
    Those projects score T-B-A- (no signal) which is the correct outcome.
    """
    bands = ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]

    def make(start, end):
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi)
              .filterDate(start, end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_SCENE_FILTER))
              .map(mask_clouds)
              .select(bands)
        )

    primary  = make(s, e)
    fallback = make(f"{s[:4]}-01-01", f"{s[:4]}-12-31")
    col      = ee.Algorithms.If(primary.size().gte(MIN_SCENES), primary, fallback)
    median   = ee.ImageCollection(col).median()

    # Empty-image guard
    empty    = ee.Image.constant([0, 0, 0, 0, 0, 0, 0]).rename(bands).toFloat()
    has_data = median.bandNames().size().gt(0)
    return ee.Image(ee.Algorithms.If(has_data, median, empty))


# ── DATA QUALITY: scene count + pixel-level composite validation ─────────────
def count_scenes(aoi, s, e):
    """
    Count cloud-free S2 SR scenes in the collection for this AOI and window.
    NOTE: This counts scenes before pixel-level masking. A composite with
    7 scenes can still be completely black if all pixels are clouded at the
    specific location. Use composite_has_data() for the definitive check.
    """
    try:
        n = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi)
              .filterDate(s, e)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_SCENE_FILTER))
              .size()
              .getInfo()
        )
        return int(n)
    except Exception:
        return 0


def composite_has_data(composite, aoi):
    """
    Check if the composite actually has non-zero pixel values over the AOI.
    This is the DEFINITIVE empty-composite check.

    count_scenes() counts scenes before pixel-level cloud masking.
    A composite with 7 scenes can still be completely black if every pixel
    at this location is masked as cloudy — a known problem for 2017 projects
    in the Philippines where S2 SR coverage was still sparse and cloudy.

    Method: sample mean of B4 (red band) over the AOI.
    - If mean is None or 0.0 → all pixels masked → black image → invalid.
    - One small reduceRegion call, negligible GEE cost.
    """
    try:
        result = (
            composite.select("B4")
              .reduceRegion(
                  reducer=ee.Reducer.mean(),
                  geometry=aoi,
                  scale=20,
                  maxPixels=1e6,
              ).getInfo()
        )
        val = result.get("B4")
        return val is not None and float(val) > 0
    except Exception:
        return False


# ── INDICES ──────────────────────────────────────────────────────────────────
def compute_indices(img):
    """All 9 spectral indices on a [0,1] scaled composite."""
    B2  = img.select("B2");  B3  = img.select("B3");  B4  = img.select("B4")
    B8  = img.select("B8");  B8A = img.select("B8A")
    B11 = img.select("B11"); B12 = img.select("B12")

    ndvi  = img.normalizedDifference(["B8",  "B4"]).rename("NDVI")
    ndbi  = img.normalizedDifference(["B11", "B8A"]).rename("NDBI")  # B8A avoids red-edge
    mndwi = img.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
    savi  = B8.subtract(B4).divide(B8.add(B4).add(0.5)).multiply(1.5).rename("SAVI")
    ibi   = (ndbi.subtract(savi.add(mndwi).divide(2))
                 .divide(ndbi.add(savi.add(mndwi).divide(2)))
                 .rename("IBI"))
    bui   = ndbi.subtract(ndvi).rename("BUI")
    ndti  = img.normalizedDifference(["B4", "B3"]).rename("NDTI")

    def tc(name, coeffs):
        out = ee.Image(0)
        for b, c in zip([B2, B3, B4, B8, B11, B12], coeffs):
            out = out.add(b.multiply(c))
        return out.rename(name)

    return img.addBands([
        ndvi, ndbi, mndwi, savi, ibi, bui, ndti,
        tc("TCB", TC["TCB"]),
        tc("TCG", TC["TCG"]),
        tc("TCW", TC["TCW"]),
    ]).select(INDEX_NAMES)


# ── TILE URLS (permanent, no expiry) ─────────────────────────────────────────
def make_tile_url(lat, lon, s, e):
    """
    Returns a permanent GEE XYZ tile URL via getMapId() instead of a signed
    thumbnail via getThumbURL().

    Why tile URLs instead of thumbnails:
    - getThumbURL() returns a pre-signed JPEG that expires in ~2-3 days.
    - getMapId() returns an XYZ tile endpoint that is permanent and public —
      no re-authentication, no re-download needed at 24k scale.
    - The URL works directly in Leaflet: L.tileLayer(url).addTo(map)
    - No hosting costs — tiles are streamed on demand from GEE.

    Trade-off: getMapId() is slightly slower per call than getThumbURL()
    because GEE pre-renders the tile pyramid server-side. Parallelism in
    generate_tiles_parallel() compensates for this.
    """
    try:
        pt  = ee.Geometry.Point([lon, lat])
        aoi = pt.buffer(THUMB_BUFFER_M).bounds()

        def tile_col(start, end, max_cloud):
            return (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
                  .map(mask_clouds)
                  .select(["B4", "B3", "B2"])
            )

        col      = tile_col(s, e, 60)
        fallback = tile_col(f"{s[:4]}-01-01", f"{s[:4]}-12-31", 80)
        col      = ee.ImageCollection(ee.Algorithms.If(col.size().gte(3), col, fallback))

        map_id = col.median().getMapId({
            "bands": ["B4", "B3", "B2"],
            "min":   0.02,
            "max":   0.25,
            "gamma": 1.4,
        })

        # url_format is the permanent {z}/{x}/{y} tile template
        return map_id["tile_fetcher"].url_format

    except Exception as exc:
        print(f"  [TILE WARN] ({lat:.4f},{lon:.4f}): {exc}")
        return None


def generate_tiles_parallel(group_rows, bs, be, as_, ae, workers):
    """
    Generate before/after tile URLs for all projects in a group in parallel.
    Each getMapId() is one HTTP request to GEE — safe to parallelise with threads.
    """
    tile_results = {}

    def fetch(row):
        pid    = str(row["project_id"])
        before = make_tile_url(row["latitude"], row["longitude"], bs, be)
        after  = make_tile_url(row["latitude"], row["longitude"], as_, ae)
        return pid, before, after

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, row): row["project_id"]
                   for _, row in group_rows.iterrows()}
        for fut in as_completed(futures):
            try:
                pid, before, after = fut.result()
                tile_results[pid] = (before, after)
            except Exception as exc:
                pid = str(futures[fut])
                print(f"  [TILE ERR] {pid}: {exc}")
                tile_results[pid] = (None, None)

    return tile_results


# ── PROCESS ONE DATE GROUP ────────────────────────────────────────────────────
def process_date_group(group, sy, cy, skip_thumbs, thumb_workers):
    """
    Process all projects sharing the same (start_year, completion_year).

    This function is the unit of parallelism — it runs in its own thread
    within a ThreadPoolExecutor. Each date group:
      1. Builds ONE pair of composites (shared by all projects in the group)
      2. Runs QA checks for before + after composites IN PARALLEL (2 threads)
      3. Runs all 3 spatial windows IN PARALLEL (3 threads, biggest speedup)
      4. Generates thumbnails in a nested ThreadPoolExecutor

    Thread safety: each nested executor writes to isolated local structures
    and returns results — no shared mutable state between threads.
    """
    bs, be  = before_window(sy)
    as_, ae = after_window(cy)

    print(f"  -> {sy}->{cy} | {len(group)} projects | "
          f"threads={thumb_workers if not skip_thumbs else 0} thumbs", flush=True)

    # Init result dict for this group
    # Note: latitude/longitude are NOT added here — they already exist in the
    # input df and are preserved through the merge in run(). Adding them here
    # causes pandas to create latitude_x/latitude_y duplicate columns.
    group_results = {
        str(r["project_id"]): {"project_id": str(r["project_id"])}
        for _, r in group.iterrows()
    }

    # AOI: union of buffered points — NOT bounding box
    # Bounding box over projects in multiple provinces = composite over entire PH
    aoi = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r["longitude"], r["latitude"]]).buffer(200))
        for _, r in group.iterrows()
    ]).union(maxError=100)

    # One composite pair shared by all projects in this date group
    before_c = build_composite(aoi, bs, be)
    after_c  = build_composite(aoi, as_, ae)

    # ── QA: before + after checks run IN PARALLEL ─────────────────────────────
    # Previously 4 serial GEE calls; now before/after fire simultaneously.
    # Stage 1: count scenes (catches pre-S2 era)
    # Stage 2: pixel-level check (catches "7 scenes but all cloudy here")
    def qa_check(composite, s, e):
        n  = count_scenes(aoi, s, e)
        ok = composite_has_data(composite, aoi) if n >= 3 else False
        return n, ok

    with ThreadPoolExecutor(max_workers=2) as qa_pool:
        f_before = qa_pool.submit(qa_check, before_c, bs, be)
        f_after  = qa_pool.submit(qa_check, after_c,  as_, ae)
        before_scenes, before_ok = f_before.result()
        after_scenes,  after_ok  = f_after.result()

    data_ok = before_ok and after_ok
    status  = "OK" if data_ok else (
        f"INVALID -- before={'OK' if before_ok else f'EMPTY ({before_scenes} scenes)'} "
        f"after={'OK' if after_ok else f'EMPTY ({after_scenes} scenes)'}"
    )
    print(f"     scenes: before={before_scenes} after={after_scenes} | {status}", flush=True)

    for _, r in group.iterrows():
        pid = str(r["project_id"])
        group_results[pid]["before_scene_count"] = before_scenes
        group_results[pid]["after_scene_count"]  = after_scenes
        group_results[pid]["composite_valid"]    = data_ok

    delta = (compute_indices(after_c)
             .subtract(compute_indices(before_c))
             .select(INDEX_NAMES))

    # ── 3 spatial windows — now run IN PARALLEL ───────────────────────────────
    # Previously sequential; each is an independent reduceRegions HTTP call.
    # Running all 3 simultaneously cuts window time by ~2/3 — the biggest
    # single speedup in this version.
    def run_window(w):
        reducer = ee.Reducer.mean() if w["reducer"] == "mean" else ee.Reducer.max()
        fc = ee.FeatureCollection([
            ee.Feature(
                ee.Geometry.Point([r["longitude"], r["latitude"]]).buffer(w["radius"]),
                {"project_id": str(r["project_id"])}
            )
            for _, r in group.iterrows()
        ])
        try:
            sampled = delta.reduceRegions(
                collection=fc,
                reducer=reducer,
                scale=20,
                tileScale=8,
            ).getInfo()["features"]
            return w["name"], sampled, None
        except Exception as exc:
            return w["name"], None, exc

    with ThreadPoolExecutor(max_workers=len(WINDOWS)) as win_pool:
        win_futures = {win_pool.submit(run_window, w): w["name"] for w in WINDOWS}
        for fut in as_completed(win_futures):
            wname, sampled, exc = fut.result()
            if exc:
                print(f"  [WARN] Window '{wname}' ({sy}->{cy}): {exc}")
                for _, r in group.iterrows():
                    pid = str(r["project_id"])
                    for idx in INDEX_NAMES:
                        group_results[pid][f"{wname}_{idx.lower()}"] = None
            else:
                for feat in sampled:
                    pid   = str(feat["properties"]["project_id"])
                    props = feat["properties"]
                    for idx in INDEX_NAMES:
                        group_results[pid][f"{wname}_{idx.lower()}"] = props.get(idx)

    # Tile URLs — permanent XYZ tiles, no expiry, stream on demand from GEE
    if not skip_thumbs:
        tiles = generate_tiles_parallel(group, bs, be, as_, ae, thumb_workers)
        for pid, (before_url, after_url) in tiles.items():
            group_results[pid]["before_tile_url"] = before_url
            group_results[pid]["after_tile_url"]  = after_url
    else:
        for _, r in group.iterrows():
            pid = str(r["project_id"])
            group_results[pid]["before_tile_url"] = None
            group_results[pid]["after_tile_url"]  = None

    return group_results


# ── PROCESS BATCH ─────────────────────────────────────────────────────────────
def process_batch(batch_df, skip_thumbs, spectral_workers, thumb_workers):
    """
    Process one batch of projects.

    Date groups within the batch run in parallel via ThreadPoolExecutor.
    This is safe because each group:
      - Builds its own composite over its own AOI (no shared GEE state)
      - Writes to its own result dict (no shared Python state)
      - Threads release the GIL during GEE HTTP requests

    spectral_workers: how many date groups run simultaneously.
    thumb_workers:    how many thumbnails generate simultaneously per group.

    Total concurrent GEE requests ≈ spectral_workers * (3 windows + thumb_workers)
    Keep spectral_workers * thumb_workers <= 20 to avoid GEE rate limit.
    """
    all_results = {}
    date_groups = list(batch_df.groupby(["start_year", "completion_year"]))

    with ThreadPoolExecutor(max_workers=spectral_workers) as pool:
        futures = {
            pool.submit(
                process_date_group,
                group, int(sy), int(cy),
                skip_thumbs, thumb_workers
            ): (sy, cy)
            for (sy, cy), group in date_groups
        }

        for fut in as_completed(futures):
            sy, cy = futures[fut]
            try:
                group_results = fut.result()
                all_results.update(group_results)
            except Exception as exc:
                print(f"  [BATCH ERR] {sy}->{cy}: {exc}")

    return list(all_results.values())


# ── SIGNAL PATTERN ────────────────────────────────────────────────────────────
def signal_pattern(row):
    THR = 0.05
    t = row.get("tight_ibi")    or 0
    b = row.get("broad_ibi")    or 0
    a = row.get("anywhere_ibi") or 0
    return (f"T{'+' if t > THR else '-'}"
            f"B{'+' if b > THR else '-'}"
            f"A{'+' if a > THR else '-'}")


# ── MAIN RUN ─────────────────────────────────────────────────────────────────
def run(input_csv, output_csv, project=None, test=False,
        skip_thumbs=False, spectral_workers=SPECTRAL_WORKERS,
        thumb_workers=THUMB_WORKERS):

    init_gee(project)

    df = pd.read_csv(input_csv)

    required = {"project_id", "latitude", "longitude", "start_year", "completion_year"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run 01_load_data.py first.")

    dropped = df[df["start_year"] < S2_MIN_YEAR]
    df      = df[df["start_year"] >= S2_MIN_YEAR].copy()
    if len(dropped):
        print(f"[SKIP] {len(dropped)} projects before {S2_MIN_YEAR} (pre-Sentinel-2 era)")

    if test:
        n_test = min(500, len(df))
        df = stratified_sample(df, n=n_test, seed=42)
        print(f"[TEST] Stratified sample: {len(df)} projects")
        print(f"       Years:   {sorted(df['completion_year'].unique())}")
        print(f"       Regions: {df['region'].nunique() if 'region' in df.columns else 'N/A'} unique")
        print(f"       (Proportional by completion_year x region — 95% CI ±4.3%)")
        skip_thumbs = False   # always generate thumbnails, even in test mode

    df["start_year"]      = df["start_year"].astype(int)
    df["completion_year"] = df["completion_year"].astype(int)

    # ── CHECKPOINT: resume from partial run if checkpoint file exists ──────────
    # Checkpoint file sits next to the output CSV, named <output>_checkpoint.csv
    checkpoint_csv = output_csv.replace(".csv", "_checkpoint.csv")
    all_results    = []
    done_ids       = set()

    if os.path.exists(checkpoint_csv):
        checkpoint_df = pd.read_csv(checkpoint_csv, dtype={"project_id": str})
        all_results   = checkpoint_df.to_dict("records")
        done_ids      = set(checkpoint_df["project_id"].astype(str))
        print(f"[RESUME] Checkpoint found: {len(done_ids)} projects already done, skipping them.")
    else:
        print(f"[CHECKPOINT] No checkpoint found — starting fresh. Progress saved to: {checkpoint_csv}")

    # Filter out already-processed projects before batching
    df_remaining = df[~df["project_id"].astype(str).isin(done_ids)].copy()

    n_date_groups = df_remaining.groupby(["start_year", "completion_year"]).ngroups
    print(f"\n[INFO] {len(df):,} total projects | {len(done_ids)} already done | {len(df_remaining):,} remaining")
    print(f"[INFO] {n_date_groups} unique date groups remaining")
    print(f"[INFO] Batch size:        {BATCH_SIZE}")
    print(f"[INFO] Spectral workers:  {spectral_workers} (parallel date groups)")
    print(f"[INFO] Thumbnail workers: {thumb_workers if not skip_thumbs else 0} {'(skipped)' if skip_thumbs else '(parallel per group)'}")
    print(f"[INFO] Max concurrent GEE requests: ~{spectral_workers * (3 + thumb_workers)}\n")

    batches = [df_remaining.iloc[i:i+BATCH_SIZE] for i in range(0, len(df_remaining), BATCH_SIZE)]

    for i, batch in enumerate(tqdm(batches, desc="Batches")):
        print(f"\n[BATCH {i+1}/{len(batches)}] {len(batch)} projects", flush=True)
        batch_results = process_batch(batch, skip_thumbs, spectral_workers, thumb_workers)
        all_results.extend(batch_results)

        # 💾 Write checkpoint after every batch so a crash never loses more than one batch
        pd.DataFrame(all_results).to_csv(checkpoint_csv, index=False)
        print(f"  [CHECKPOINT] {len(all_results)}/{len(df)} projects saved → {checkpoint_csv}", flush=True)

    results_df = pd.DataFrame(all_results)

    # Drop any lat/lon columns that crept into results_df to prevent
    # pandas creating latitude_x / latitude_y suffixed duplicates.
    results_df = results_df.drop(columns=["latitude","longitude"], errors="ignore")

    output_df  = df.merge(results_df, on="project_id", how="left")

    # Confirm lat/lon survived the merge
    assert "latitude"  in output_df.columns, "latitude column missing after merge"
    assert "longitude" in output_df.columns, "longitude column missing after merge"

    output_df["signal_pattern"] = output_df.apply(
        lambda r: signal_pattern(r.to_dict()), axis=1
    )
    output_df["signal_interpretation"] = output_df["signal_pattern"].map(
        lambda p: PATTERN_MEANINGS.get(p, "Unknown -- manual review needed")
    )

    output_df.to_csv(output_csv, index=False)

    # ── CHECKPOINT CLEANUP: remove checkpoint now that final output is written ──
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)
        print(f"[CHECKPOINT] Removed {checkpoint_csv} (run complete)")

    print(f"\n[DONE] -> {output_csv}")
    print(f"[INFO] {len(output_df):,} rows saved")
    print(f"\n[SIGNAL PATTERNS]")
    print(output_df["signal_pattern"].value_counts().to_string())
    print(f"\nNext: python 03_score.py --input {output_csv} --output results_scored.csv")


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bantay ghost project GEE pipeline")
    p.add_argument("--input",            required=True,  help="CSV from 01_load_data.py")
    p.add_argument("--output",           required=True,  help="Output CSV with deltas")
    p.add_argument("--project",          default=None,   help="GEE cloud project ID")
    p.add_argument("--test",             action="store_true", help="Run on first 50 rows only")
    p.add_argument("--skip-thumbs",      action="store_true", help="Skip thumbnail generation (faster)")
    p.add_argument("--spectral-workers", type=int, default=SPECTRAL_WORKERS,
                   help=f"Parallel date groups (default: {SPECTRAL_WORKERS})")
    p.add_argument("--thumb-workers",    type=int, default=THUMB_WORKERS,
                   help=f"Parallel thumbnails per group (default: {THUMB_WORKERS})")
    a = p.parse_args()

    run(
        a.input, a.output, a.project, a.test,
        a.skip_thumbs,
        a.spectral_workers,
        a.thumb_workers,
    )