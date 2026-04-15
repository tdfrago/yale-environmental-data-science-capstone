"""
0_data_collection.py
====================
Downloads the DPWH transparency dataset from HuggingFace and saves it
as a GeoPackage (.gpkg) for use in the GIST pipeline.

Source dataset:
    bettergovph/dpwh-transparency-data
    https://huggingface.co/datasets/bettergovph/dpwh-transparency-data
    License: CC0 1.0

Output:
    dpwh_data.gpkg  --  all DPWH project records with point geometry

Usage:
    python 0_data_collection.py
    python 0_data_collection.py --output data/raw/dpwh_data.gpkg
"""

import argparse
from datasets import load_dataset
import geopandas as gpd


def main(output_path):
    print("[DOWNLOAD] Fetching bettergovph/dpwh-transparency-data from HuggingFace...")
    dataset = load_dataset("bettergovph/dpwh-transparency-data")

    df = dataset["train"].to_pandas()
    print(f"[INFO] {len(df):,} records loaded")
    print(f"[INFO] Columns: {list(df.columns)}")

    print("[GEO] Building GeoDataFrame from latitude/longitude columns...")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )

    print(f"[SAVE] Writing to {output_path}...")
    gdf.to_file(output_path, layer="dpwh_projects", driver="GPKG")
    print(f"[DONE] Saved {len(gdf):,} records to {output_path}")
    print(f"\nNext step:")
    print(f"  python scripts/01_load_data.py --input {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dpwh_data.gpkg",
        help="Output path for the GeoPackage file (default: dpwh_data.gpkg)"
    )
    args = parser.parse_args()
    main(args.output)
