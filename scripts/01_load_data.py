"""
01_load_data.py
===============
Load the DPWH GeoPackage → validate coordinates → export pipeline-ready CSVs.

Usage:
    python 01_load_data.py --input "C:\\Users\\frago\\Desktop\\yale\\dpwh_data.gpkg"

Outputs:
    all_projects.csv       All records flattened to CSV
    flood_projects.csv     Flood control only, ready for pipeline.py
    coord_check.csv        Projects with suspicious coordinates (for review)
"""

import argparse
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Only categories we want for the ghost project pipeline
FLOOD_KEYWORDS = ["flood", "drainage", "riverbank", "river control",
                  "embankment", "seawall", "slope protection"]

# Statuses to include (exclude On-Going — no completion date yet)
VALID_STATUSES = ["Completed"]

# Sentinel-2 SR starts mid-2017; only projects starting from this year
MIN_START_YEAR = 2017

# Philippine bounding box — coordinates outside this are clearly wrong
PH_BBOX = {"lat_min": 4.5, "lat_max": 21.5,
           "lon_min": 116.0, "lon_max": 127.0}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def unpack_location(val):
    """Unpack the location column which may be a dict, JSON string, or None."""
    if isinstance(val, dict):
        return val.get("province", ""), val.get("region", "")
    if isinstance(val, str):
        try:
            d = json.loads(val)
            return d.get("province", ""), d.get("region", "")
        except Exception:
            return "", ""
    return "", ""


def extract_coords_from_geometry(gdf):
    """
    If the GeoPackage has a real geometry column, extract lat/lon from it.
    Falls back to existing latitude/longitude columns if present.
    """
    if gdf.geometry is not None and not gdf.geometry.is_empty.all():
        # Reproject to WGS84 if needed
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf["longitude"] = gdf.geometry.centroid.x
        gdf["latitude"]  = gdf.geometry.centroid.y
        print("[INFO] Coordinates extracted from geometry column.")
    else:
        # Geometry column missing or empty — use existing lat/lon columns
        print("[INFO] No geometry found — using latitude/longitude columns.")
    return gdf


def flag_suspect_coordinates(df):
    """
    Three coordinate sanity checks:
    1. Outside Philippine bounding box (clearly wrong)
    2. Exactly (0, 0) — null island
    3. latitude and longitude are swapped (lat > 90 means it's actually lon)
    """
    b = PH_BBOX
    outside_ph  = (
        (df["latitude"]  < b["lat_min"]) | (df["latitude"]  > b["lat_max"]) |
        (df["longitude"] < b["lon_min"]) | (df["longitude"] > b["lon_max"])
    )
    null_island = (df["latitude"] == 0) & (df["longitude"] == 0)
    swapped     = df["latitude"] > 90   # lat should never exceed 90

    df["coord_flag"] = "ok"
    df.loc[outside_ph,  "coord_flag"] = "outside_philippines"
    df.loc[null_island, "coord_flag"] = "null_island"
    df.loc[swapped,     "coord_flag"] = "lat_lon_swapped"
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(input_path: str):
    print(f"\n[LOAD] Reading: {input_path}")
    gdf = gpd.read_file(input_path)
    print(f"[INFO] Raw records: {len(gdf):,}")
    print(f"[INFO] Columns:     {list(gdf.columns)}")

    # ── Convert to regular DataFrame ─────────────────────────────────────────
    gdf = extract_coords_from_geometry(gdf)
    df  = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

    # ── Unpack location dict column ───────────────────────────────────────────
    if "location" in df.columns:
        df[["province", "region"]] = df["location"].apply(
            lambda v: pd.Series(unpack_location(v))
        )
    else:
        # Try to infer from existing columns
        for col in ["province", "region"]:
            if col not in df.columns:
                df[col] = ""

    # ── Standardise column names ──────────────────────────────────────────────
    rename_map = {
        "contractId":     "project_id",
        "description":    "project_name",
        "budget":         "contract_amount",
        "category":       "category",
        "status":         "status",
        "contractor":     "contractor",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Parse dates → years ───────────────────────────────────────────────────
    df["start_year"]      = pd.to_datetime(df.get("startDate"),      errors="coerce").dt.year.astype("Int64")
    df["completion_year"] = pd.to_datetime(df.get("completionDate"), errors="coerce").dt.year.astype("Int64")

    # ── Ensure lat/lon are numeric ────────────────────────────────────────────
    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

    # ── Coordinate sanity flags ───────────────────────────────────────────────
    df = flag_suspect_coordinates(df)
    suspect = df[df["coord_flag"] != "ok"]
    print(f"\n[COORD CHECK]")
    print(df["coord_flag"].value_counts().to_string())

    # ── Export all projects ───────────────────────────────────────────────────
    all_cols = [
        "project_id", "project_name", "category", "status",
        "latitude", "longitude", "coord_flag",
        "start_year", "completion_year",
        "contract_amount", "contractor",
        "province", "region",
    ]
    existing_cols = [c for c in all_cols if c in df.columns]
    df[existing_cols].to_csv("all_projects.csv", index=False)
    print(f"\n[SAVED] all_projects.csv  ({len(df):,} rows)")

    # ── Export suspect coordinates for manual review ──────────────────────────
    suspect[existing_cols].to_csv("coord_check.csv", index=False)
    print(f"[SAVED] coord_check.csv   ({len(suspect):,} rows — review these)")

    # ── Filter for pipeline ───────────────────────────────────────────────────
    is_flood = df["category"].str.lower().str.contains(
        "|".join(FLOOD_KEYWORDS), na=False
    )
    is_complete  = df["status"].isin(VALID_STATUSES)
    has_coords   = df["latitude"].notna() & df["longitude"].notna()
    coords_ok    = df["coord_flag"] == "ok"
    in_s2_era    = df["start_year"] >= MIN_START_YEAR
    has_end_year = df["completion_year"].notna()

    flood_df = df[
        is_flood & is_complete & has_coords & coords_ok & in_s2_era & has_end_year
    ][existing_cols].reset_index(drop=True)

    flood_df.to_csv("flood_projects.csv", index=False)
    print(f"[SAVED] flood_projects.csv ({len(flood_df):,} rows — pipeline-ready)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Total records:              {len(df):,}")
    print(f"Flood/drainage projects:    {is_flood.sum():,}")
    print(f"Status = Completed:         {is_complete.sum():,}")
    print(f"Coordinates OK:             {(has_coords & coords_ok).sum():,}")
    print(f"Within Sentinel-2 era:      {in_s2_era.sum():,}")
    print(f"PIPELINE-READY (flood):     {len(flood_df):,}")
    print(f"\nNext step:")
    print(f"  python 02_pipeline.py --input flood_projects.csv --output results.csv --test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .gpkg file")
    args = parser.parse_args()
    main(args.input)
