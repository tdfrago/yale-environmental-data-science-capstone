"""
04_upload.py
============
1. Generates GEE before/after thumbnail URLs for each project
2. Uploads everything to Supabase in batches

Run the SQL at the bottom of this file in your Supabase SQL editor FIRST,
then run this script.

Usage:
    python 04_upload.py \\
        --input results_scored.csv \\
        --url https://YOUR_PROJECT.supabase.co \\
        --key YOUR_SERVICE_ROLE_KEY

Args:
    --skip-thumbs   Skip thumbnail generation if you already have them
    --gee-project   GEE cloud project ID (optional)

---- SUPABASE SCHEMA — paste into Supabase SQL editor ----

create table projects (
  project_id              text primary key,
  project_name            text,
  province                text,
  region                  text,
  latitude                float,
  longitude               float,
  start_year              int,
  completion_year         int,
  contract_amount         float,
  contractor              text,
  category                text,
  status                  text,

  -- Scoring output
  ghost_probability       float,
  ghost_tier              text,
  signal_pattern          text,
  signal_interpretation   text,
  coord_displacement      text,
  neighbour_mean_ibi      float,
  anomalous_vs_neighbourhood bool,

  -- GEE thumbnails
  before_thumb_url        text,
  after_thumb_url         text,

  -- Spectral deltas — tight window (mean @ 30m)
  tight_ibi   float, tight_bui   float, tight_ndvi  float, tight_ndbi  float,
  tight_mndwi float, tight_tcb   float, tight_tcg   float, tight_tcw   float,
  tight_ndti  float,

  -- Spectral deltas — broad window (mean @ 100m)
  broad_ibi   float, broad_bui   float, broad_ndvi  float, broad_ndbi  float,
  broad_mndwi float, broad_tcb   float, broad_tcg   float, broad_tcw   float,
  broad_ndti  float,

  -- Spectral deltas — anywhere window (max @ 150m)
  anywhere_ibi   float, anywhere_bui   float, anywhere_ndvi  float,
  anywhere_ndbi  float, anywhere_mndwi float, anywhere_tcb   float,
  anywhere_tcg   float, anywhere_tcw   float, anywhere_ndti  float
);

-- Read-only access for the website (anon key is safe to expose)
alter table projects enable row level security;
create policy "public read" on projects for select using (true);

-- Speed up common filter patterns
create index on projects (ghost_probability desc);
create index on projects (completion_year);
create index on projects (ghost_tier);
create index on projects (coord_displacement);
create index on projects using gin(
  to_tsvector('english',
    coalesce(project_name,'') || ' ' ||
    coalesce(province,'')     || ' ' ||
    coalesce(region,'')
  )
);

-----------------------------------------------------------
"""

import ee
import pandas as pd
import argparse
import time
from tqdm import tqdm
from supabase import create_client

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 500   # rows per Supabase upsert
GEE_BATCH    = 30    # projects per thumbnail batch (GEE memory limit)
THUMB_SIZE   = 256   # pixels — small enough to be fast, clear enough to use
THUMB_BANDS  = ["B4", "B3", "B2"]   # true colour RGB
THUMB_GAMMA  = 1.4

DRY_START = "11-01"
DRY_END   = "04-30"
CLOUD_FLT = 60


# ── GEE THUMBNAIL HELPERS ─────────────────────────────────────────────────────

def get_composite_for_thumb(lat, lon, start_date, end_date):
    """Build a small true-colour composite for thumbnail display."""
    pt  = ee.Geometry.Point([lon, lat]).buffer(300).bounds()
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(pt)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FLT))
          .map(lambda img: img.divide(10000)
               .updateMask(img.select("SCL").neq(9).And(img.select("SCL").neq(3)))
               .copyProperties(img, ["system:time_start"]))
          .median()
          .select(THUMB_BANDS)
    )
    return col


def make_thumb_url(lat, lon, start_date, end_date):
    """Return a signed JPEG thumbnail URL from GEE."""
    try:
        img = get_composite_for_thumb(lat, lon, start_date, end_date)
        pt  = ee.Geometry.Point([lon, lat]).buffer(500).bounds()
        url = img.getThumbURL({
            "bands":      THUMB_BANDS,
            "region":     pt,
            "dimensions": THUMB_SIZE,
            "gamma":      THUMB_GAMMA,
            "format":     "jpg",
        })
        return url
    except Exception as exc:
        print(f"\n[WARN] Thumbnail failed ({lat},{lon}): {exc}")
        return None


def generate_thumbnails(batch_df):
    """Generate before/after URLs for a batch of projects."""
    results = {}
    for _, row in batch_df.iterrows():
        pid = str(row["project_id"])
        sy  = int(row["start_year"])
        cy  = int(row["completion_year"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        bs = f"{sy-1}-{DRY_START}";  be = f"{sy}-{DRY_END}"
        as_= f"{cy}-{DRY_START}";   ae = f"{cy+1}-{DRY_END}"

        results[pid] = (
            make_thumb_url(lat, lon, bs, be),
            make_thumb_url(lat, lon, as_, ae),
        )
        time.sleep(0.1)
    return results


# ── SUPABASE UPLOAD ───────────────────────────────────────────────────────────

# All columns the schema accepts
SCHEMA_COLS = [
    "project_id", "project_name", "province", "region",
    "latitude", "longitude", "start_year", "completion_year",
    "contract_amount", "contractor", "category", "status",
    "ghost_probability", "ghost_tier", "signal_pattern",
    "signal_interpretation", "coord_displacement",
    "neighbour_mean_ibi", "anomalous_vs_neighbourhood",
    "before_thumb_url", "after_thumb_url",
    "before_scene_count", "after_scene_count",   # data quality flags
    "composite_valid", "data_quality",            # data quality flags
    # tight
    "tight_ibi","tight_bui","tight_ndvi","tight_ndbi",
    "tight_mndwi","tight_tcb","tight_tcg","tight_tcw","tight_ndti",
    # broad
    "broad_ibi","broad_bui","broad_ndvi","broad_ndbi",
    "broad_mndwi","broad_tcb","broad_tcg","broad_tcw","broad_ndti",
    # anywhere
    "anywhere_ibi","anywhere_bui","anywhere_ndvi","anywhere_ndbi",
    "anywhere_mndwi","anywhere_tcb","anywhere_tcg","anywhere_tcw","anywhere_ndti",
]


def upload(df, client, table="projects"):
    """Upsert rows to Supabase in batches."""
    import numpy as np

    # Only send columns that exist in both the dataframe and the schema
    cols      = [c for c in SCHEMA_COLS if c in df.columns]
    upload_df = df[cols].copy()
    upload_df.columns = [c.lower() for c in upload_df.columns]

    # Replace inf/-inf with NaN, then NaN -> None
    # JSON cannot encode inf, -inf, or NaN -- Supabase needs Python None (null)
    upload_df = upload_df.replace([np.inf, -np.inf], np.nan)
    upload_df = upload_df.astype(object).where(pd.notnull(upload_df), None)

    # Final pass: catch any remaining non-JSON-compliant floats
    def clean_row(row):
        cleaned = {}
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned

    rows    = [clean_row(r) for r in upload_df.to_dict(orient="records")]
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Uploading to Supabase"):
        client.table(table).upsert(batch).execute()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(input_csv, supabase_url, supabase_key,
        gee_project=None, table="projects", skip_thumbs=False):

    # Init GEE
    if not skip_thumbs:
        try:
            ee.Initialize(project=gee_project) if gee_project else ee.Initialize()
            print("[GEE] ✓ Ready for thumbnails")
        except Exception as e:
            print(f"[GEE] ✗ {e} — run with --skip-thumbs to skip thumbnail generation")
            raise

    # Init Supabase
    client = create_client(supabase_url, supabase_key)
    print("[Supabase] ✓ Connected")

    df = pd.read_csv(input_csv)
    print(f"[INFO] {len(df):,} projects loaded")

    # ── Thumbnail generation ──────────────────────────────────────────────────
    if not skip_thumbs and "before_thumb_url" not in df.columns:
        print(f"[INFO] Generating thumbnails in batches of {GEE_BATCH}...")
        thumbs = {}
        batches = [df.iloc[i:i+GEE_BATCH] for i in range(0, len(df), GEE_BATCH)]
        for batch in tqdm(batches, desc="GEE thumbnails"):
            thumbs.update(generate_thumbnails(batch))

        df["before_thumb_url"] = df["project_id"].astype(str).map(
            lambda pid: thumbs.get(pid, (None, None))[0]
        )
        df["after_thumb_url"]  = df["project_id"].astype(str).map(
            lambda pid: thumbs.get(pid, (None, None))[1]
        )

    # ── Upload ────────────────────────────────────────────────────────────────
    print(f"[INFO] Uploading to table '{table}'...")
    upload(df, client, table)
    print(f"[DONE] ✓ {len(df):,} rows upserted to Supabase")
    print(f"\nNext step: configure index.html with your Supabase URL + anon key,")
    print(f"then push to GitHub and enable Pages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--url",         required=True,  help="Supabase project URL")
    parser.add_argument("--key",         required=True,  help="Supabase service role key")
    parser.add_argument("--gee-project", default=None)
    parser.add_argument("--table",       default="projects")
    parser.add_argument("--skip-thumbs", action="store_true")
    args = parser.parse_args()
    run(args.input, args.url, args.key, args.gee_project, args.table, args.skip_thumbs)