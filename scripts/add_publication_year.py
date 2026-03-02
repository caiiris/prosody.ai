"""
Add a Year column to the poetry dataset using a poet -> year mapping.
Outputs a NEW file with all poems preserved; poems whose poet is missing from
the mapping will have a blank Year.
"""
import sys
from pathlib import Path

import pandas as pd

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POEMS_CSV = DATA_DIR / "PoetryFoundationData.csv"
MAPPING_CSV = DATA_DIR / "poet_year_mapping.csv"
OUTPUT_CSV = DATA_DIR / "PoetryFoundationData_with_year.csv"


def main():
    import argparse
    p = argparse.ArgumentParser(description="Add Year column from poet_year_mapping.")
    p.add_argument("--limit", type=int, default=None, help="Only process first N poems (for testing).")
    args = p.parse_args()

    if not MAPPING_CSV.exists():
        print(f"Mapping file not found: {MAPPING_CSV}")
        print("Create it with columns: Poet, Year (one year per poet, e.g. publication year)")
        sys.exit(1)

    print("Loading poems...")
    poems = pd.read_csv(POEMS_CSV)
    if args.limit:
        poems = poems.head(args.limit)
        print(f"Limited to first {args.limit} poems.")
    poems["Poet"] = poems["Poet"].astype(str).str.strip()

    print("Loading poet -> year mapping...")
    mapping = pd.read_csv(MAPPING_CSV)
    mapping["Poet"] = mapping["Poet"].astype(str).str.strip()
    mapping["Year"] = pd.to_numeric(mapping["Year"], errors="coerce")
    mapping = mapping.dropna(subset=["Year"]).drop_duplicates(subset=["Poet"], keep="first")

    before = len(poems)
    merged = poems.merge(mapping[["Poet", "Year"]], on="Poet", how="left")
    after = len(merged)
    missing = int(merged["Year"].isna().sum())

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {after} poems (with year) to {OUTPUT_CSV}")
    print(f"Missing Year for {missing} poems (poet not in mapping).")


if __name__ == "__main__":
    main()
