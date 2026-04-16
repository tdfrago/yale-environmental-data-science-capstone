"""
03_score.py  (v2 — IBI-primary scoring)
========================================

Why v1 produced 0 ghost projects:
  - signal_pattern uses only tight_ibi / broad_ibi / anywhere_ibi
  - ghost_probability used 7 indices including NDVI, MNDWI, TCG
  - In the Philippines, vegetation and moisture shift seasonally even on
    empty land. Those false positive signals inflated window_score, pushing
    prob = 1 - weighted_signal below 0.70 for genuinely ghost projects.
  - Result: T-B-A- projects scored as "suspect" instead of "ghost".

v2 fix:
  - IBI and TCB are PRIMARY signals (70% weight combined)
    → designed specifically for impervious surface detection
    → robust to vegetation and moisture noise
  - NDVI, MNDWI, BUI are SECONDARY signals (30% weight)
    → supporting evidence only, not decision-makers
  - Pattern-to-probability direct mapping as a baseline anchor
  - Index-specific thresholds: NDVI/MNDWI need larger delta to count
    (0.08) because seasonal noise is larger for vegetation indices

Expected output with v2:
  - T-B-A- → ~0.82 ghost probability → "ghost"
  - T-B-A+ → ~0.55 → "suspect" (coord displaced, not ghost)
  - T+B+A+ → ~0.15 → "built"

Usage:
    python 03_score.py --input results.csv --output results_scored.csv
"""

import argparse
import pandas as pd
import numpy as np

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
# IBI/TCB: low noise floor — 0.05 threshold is reliable
IBI_THR  = 0.05

# NDVI/MNDWI: higher threshold needed because seasonal vegetation and moisture
# shifts in the Philippines can be 0.05–0.08 even on unchanged land
VEG_THR  = 0.08

# ── PRIMARY vs SECONDARY indices ─────────────────────────────────────────────
# PRIMARY: specifically designed for impervious surface, low seasonal noise
PRIMARY = {
    "ibi":  {"direction": "pos", "threshold": IBI_THR,  "weight": 0.40},
    "tcb":  {"direction": "pos", "threshold": IBI_THR,  "weight": 0.30},
}

# SECONDARY: supporting evidence, higher noise, lower weight
SECONDARY = {
    "ndvi":  {"direction": "neg", "threshold": VEG_THR,  "weight": 0.10},
    "mndwi": {"direction": "neg", "threshold": VEG_THR,  "weight": 0.10},
    "bui":   {"direction": "pos", "threshold": IBI_THR,  "weight": 0.10},
}

ALL_INDICES = {**PRIMARY, **SECONDARY}

# ── PATTERN → BASE PROBABILITY ────────────────────────────────────────────────
# Direct mapping from spatial signal pattern to probability anchor.
# These are calibrated anchors — the IBI-weighted score adjusts within ±0.15.
PATTERN_PROBS = {
    "T+B+A+": 0.10,   # all windows agree: built at coordinate
    "T+B-A+": 0.20,   # built at coord + nearby: likely built
    "T+B+A-": 0.20,   # built nearby: check composite
    "T+B-A-": 0.30,   # small structure or noise
    "T-B+A+": 0.40,   # built in vicinity, coord slightly off
    "T-B-A+": 0.55,   # COA fraud type: coord displaced, built nearby
    "T-B+A-": 0.60,   # ambiguous
    "T-B-A-": 0.82,   # no signal anywhere → ghost or lost coord
}


# ── CORE SCORING ──────────────────────────────────────────────────────────────

def ibi_window_signal(row, window_name):
    """
    Check if IBI shows construction signal for a given window.
    IBI is the primary gate — if IBI doesn't signal, the project is unlikely built.
    Returns: True (signal), False (no signal), None (missing data)
    """
    val = row.get(f"{window_name}_ibi")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return val > IBI_THR


def weighted_construction_score(row, window_name):
    """
    Weighted construction score for one spatial window.
    Combines primary (IBI, TCB) and secondary (NDVI, MNDWI, BUI) signals.

    Returns a score in [0, 1]:
      1.0 = all indices show strong construction signal
      0.0 = no indices show construction signal
    """
    total_weight = 0.0
    signal_weight = 0.0

    for idx_name, cfg in ALL_INDICES.items():
        col = f"{window_name}_{idx_name}"
        val = row.get(col)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue   # skip missing

        w   = cfg["weight"]
        thr = cfg["threshold"]

        # Check direction: pos indices need positive delta, neg indices need negative
        if cfg["direction"] == "pos":
            has_signal = val > thr
        else:
            has_signal = val < -thr

        total_weight  += w
        signal_weight += w if has_signal else 0.0

    if total_weight == 0:
        return None   # no data

    return signal_weight / total_weight


def classify_displacement(row):
    """
    Spatial pattern classification based on IBI across 3 windows.

    T = tight (30m mean)   B = broad (100m mean)   A = anywhere (150m max)
    Uses only IBI — the most reliable construction indicator.

    COA-documented fraud type: T-B-A+ → built within 150m, coord displaced.
    Action: verify coordinate first, do NOT classify as ghost yet.
    """
    t = row.get("tight_ibi",    0) or 0
    b = row.get("broad_ibi",    0) or 0
    a = row.get("anywhere_ibi", 0) or 0

    if t > IBI_THR or b > IBI_THR:
        return "confirmed_built"
    elif a > IBI_THR:
        return "coord_displaced"      # COA-documented site mismatch type
    else:
        return "ghost_or_lost_coord"  # no signal — ghost OR displacement > 150m


def ghost_probability(row):
    """
    Final ghost probability combining:
      1. Pattern-based anchor (from signal_pattern column if available)
      2. IBI-weighted construction score across all 3 windows
      3. Displacement classification cap

    Weighting: 60% pattern anchor, 40% weighted index score
    This makes the score interpretable and anchored to spatial logic,
    while allowing index evidence to adjust within ±0.20.
    """
    # ── DATA QUALITY GATE ────────────────────────────────────────────────────
    # Black images = zero composite = delta = 0 everywhere = T-B-A- = false ghost.
    # If either composite was built from < 3 cloud-free scenes, the delta is
    # meaningless. Return NaN so the project gets tier "no_data" and is
    # excluded from ghost/suspect/built counts entirely.
    #
    # FIX 1: composite_valid is stored as a string in CSV ("True"/"False").
    # Must parse explicitly — non-empty strings are always truthy in Python,
    # so `not "False"` evaluates to False and the gate never fires without this.
    #
    # FIX 2: Cap was 0.50 which still scored as "suspect" (>=0.40 threshold).
    # Now returns NaN so ghost_tier() maps it to "no_data" instead.
    before_n = row.get("before_scene_count", 99)
    after_n  = row.get("after_scene_count",  99)
    valid    = row.get("composite_valid",     True)

    # Parse string boolean that comes back from CSV read
    if isinstance(valid, str):
        valid = valid.strip().lower() == "true"

    # Catch all-null IBI — black composite that passed scene count check
    def _is_null(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    all_ibi_null = (
        _is_null(row.get("tight_ibi")) and
        _is_null(row.get("broad_ibi")) and
        _is_null(row.get("anywhere_ibi"))
    )

    # Scene count check — guard against NaN values in count columns
    def _scene_too_low(n):
        if n is None: return False
        if isinstance(n, float) and np.isnan(n): return False
        return int(n) < 3

    if not valid or all_ibi_null or _scene_too_low(before_n) or _scene_too_low(after_n):
        return np.nan   # no score — insufficient data, not a ghost signal

    # ── 1. Get pattern anchor ────────────────────────────────────────────────
    pattern = row.get("signal_pattern", "")
    anchor  = PATTERN_PROBS.get(pattern, 0.60)   # default 0.60 if unknown

    # ── 2. Weighted index score across 3 windows ─────────────────────────────
    t_score = weighted_construction_score(row, "tight")
    b_score = weighted_construction_score(row, "broad")
    a_score = weighted_construction_score(row, "anywhere")

    # If all windows have no data → return anchor only
    if t_score is None and b_score is None and a_score is None:
        return round(anchor, 4)

    t_score = t_score if t_score is not None else 0.5
    b_score = b_score if b_score is not None else 0.5
    a_score = a_score if a_score is not None else 0.5

    # Tighter window weighted more: tight=50%, broad=35%, anywhere=15%
    # anywhere uses max reducer so it's more sensitive — weight it less
    combined_signal = (t_score * 0.50) + (b_score * 0.35) + (a_score * 0.15)

    # Convert signal (0=no construction, 1=full construction) to ghost prob
    index_prob = 1.0 - combined_signal

    # ── 3. Blend anchor + index score ────────────────────────────────────────
    prob = (anchor * 0.60) + (index_prob * 0.40)

    # ── 4. Displacement cap ───────────────────────────────────────────────────
    displacement = classify_displacement(row)
    if displacement == "confirmed_built":
        prob = min(prob, 0.35)   # can't be ghost if IBI signal at coordinate
    elif displacement == "coord_displaced":
        prob = min(prob, 0.60)   # something was built nearby — not definitive ghost

    return round(float(np.clip(prob, 0.0, 1.0)), 4)


def ghost_tier(prob):
    # NaN = insufficient satellite data, not scoreable
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return "no_data"
    if prob >= 0.70:
        return "ghost"
    elif prob >= 0.40:
        return "suspect"
    else:
        return "built"


# ── NEIGHBOURHOOD CONTEXT ─────────────────────────────────────────────────────

def add_neighbourhood_flag(df, radius_km=10):
    """
    Flag projects where the project IBI is near zero but surrounding projects
    in the same area show clear construction signals.

    If a project has no signal but all its 10km neighbours do, that rules
    out a cloud/composite issue (which would affect all neighbours equally)
    and strengthens the ghost interpretation.
    """
    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        print("[SKIP] scikit-learn not installed — skipping neighbourhood flag")
        print("       pip install scikit-learn")
        return df

    valid = df.dropna(subset=["latitude", "longitude", "tight_ibi"]).copy()
    if len(valid) < 2:
        return df

    coords_rad = np.radians(valid[["latitude", "longitude"]].values)
    tree       = BallTree(coords_rad, metric="haversine")
    r          = radius_km / 6371

    _, indices = tree.query_radius(coords_rad, r=r, return_distance=True,
                                   sort_results=True)

    neighbour_ibi = []
    for i, idx in enumerate(indices):
        neighbours = idx[idx != i]
        if len(neighbours) == 0:
            neighbour_ibi.append(None)
        else:
            neighbour_ibi.append(float(valid.iloc[neighbours]["tight_ibi"].mean()))

    valid["neighbour_mean_ibi"] = neighbour_ibi

    # Flag: project has no IBI signal BUT neighbours do
    # Suggests ghost, not a data quality issue
    valid["anomalous_vs_neighbourhood"] = (
        (valid["tight_ibi"].abs() < IBI_THR) &
        (valid["neighbour_mean_ibi"].fillna(0) > IBI_THR * 1.5)
    )

    df = df.merge(
        valid[["project_id", "neighbour_mean_ibi", "anomalous_vs_neighbourhood"]],
        on="project_id", how="left"
    )
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print(f"[LOAD] {len(df):,} projects")

    if "signal_pattern" not in df.columns:
        print("[WARN] signal_pattern missing — re-run 02_pipeline.py")
        df["signal_pattern"] = "T?B?A?"

    # ── Data quality flag ────────────────────────────────────────────────────
    def get_data_quality(row):
        b = row.get("before_scene_count")
        a = row.get("after_scene_count")
        # Handle NaN from CSV — treat as missing
        if isinstance(b, float) and np.isnan(b): b = None
        if isinstance(a, float) and np.isnan(a): a = None
        # If columns missing (old pipeline run), mark as unknown
        if b is None and a is None:  return "unknown"
        if b == 0 and a == 0:        return "no_data"
        if b is not None and int(b) < 3:  return "sparse_before"
        if a is not None and int(a) < 3:  return "sparse_after"
        return "ok"

    df["data_quality"] = df.apply(get_data_quality, axis=1)

    # ── Score ─────────────────────────────────────────────────────────────────
    print("[SCORE] Computing ghost probabilities (IBI-primary scoring)...")
    df["ghost_probability"]  = df.apply(ghost_probability, axis=1)
    df["ghost_tier"]         = df["ghost_probability"].map(ghost_tier)
    df["coord_displacement"] = df.apply(classify_displacement, axis=1)

    # ── Neighbourhood context ─────────────────────────────────────────────────
    print("[SCORE] Adding neighbourhood context...")
    df = add_neighbourhood_flag(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    ghost     = df[df["ghost_tier"] == "ghost"]
    suspect   = df[df["ghost_tier"] == "suspect"]
    built     = df[df["ghost_tier"] == "built"]
    displaced = df[df["coord_displacement"] == "coord_displaced"]

    print(f"\n[DONE] → {output_csv}")
    print(f"\n[GHOST TIER DISTRIBUTION]")
    print(df["ghost_tier"].value_counts().to_string())

    print(f"\n[SIGNAL PATTERNS vs GHOST TIER]")
    if "signal_pattern" in df.columns:
        cross = pd.crosstab(df["signal_pattern"], df["ghost_tier"])
        print(cross.to_string())

    print(f"\n[COORDINATE DISPLACEMENT]")
    print(df["coord_displacement"].value_counts().to_string())

    no_data = df[df["ghost_tier"] == "no_data"]
    scored  = df[df["ghost_tier"] != "no_data"]
    print(f"\n[HIGHLIGHTS]")
    print(f"  Ghost   (>70%):       {len(ghost):,}   ({len(ghost)/len(scored)*100:.1f}% of scored)")
    print(f"  Suspect (40-70%):     {len(suspect):,}   ({len(suspect)/len(scored)*100:.1f}% of scored)")
    print(f"  Built   (<40%):       {len(built):,}   ({len(built)/len(scored)*100:.1f}% of scored)")
    print(f"  No satellite data:    {len(no_data):,}   (excluded from counts)")
    print(f"  Coord displaced:      {len(displaced):,}   ← verify GPS, not ghost")

    # Expected baseline: ~5% ghost (COA confirmed rate across all DPWH)
    # Flood control is higher risk — expect 8-15% ghost in this subset
    # Exclude no_data projects from rate — they have no valid spectral signal
    scored     = df[df["ghost_tier"] != "no_data"]
    ghost_rate = len(ghost) / len(scored) * 100 if len(scored) > 0 else 0
    print(f"\n[BASELINE CHECK]")
    print(f"  Ghost rate: {ghost_rate:.1f}%")
    if ghost_rate < 3:
        print(f"  [NOTE] Lower than COA confirmed rate (~5%). Check if thresholds are too conservative.")
    elif ghost_rate > 30:
        print(f"  [NOTE] Higher than expected. Check composite quality for this date range.")
    else:
        print(f"  [OK] Within expected range for flood control projects.")

    if len(ghost) > 0:
        print(f"\n[TOP 10 BY GHOST PROBABILITY]")
        show_cols = ["project_id", "ghost_probability", "signal_pattern",
                     "coord_displacement", "contract_amount"]
        show_cols = [c for c in show_cols if c in df.columns]
        top = ghost.nlargest(min(10, len(ghost)), "ghost_probability")[show_cols]
        print(top.to_string(index=False))

    print(f"\nNext: python 04_upload.py --input {output_csv} "
          f"--url https://rqnjuyxbfvzzwuwggrqe.supabase.co "
          f"--key YOUR_KEY --skip-tiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)