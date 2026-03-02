"""
Fetch birth year (or first known year) for each poet from Wikipedia.
Builds data/poet_year_mapping.csv so add_publication_year.py can run.
Uses birth year + 30 as proxy for "typical publication year" when exact year unknown.
Much faster than per-poem scraping: one lookup per unique poet (~1–2 hours for full dataset).
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POEMS_CSV = DATA_DIR / "PoetryFoundationData.csv"
MAPPING_CSV = DATA_DIR / "poet_year_mapping.csv"
CACHE_CSV = DATA_DIR / "poet_year_cache.csv"  # partial results for resume

USER_AGENT = "PoetryEraProject/1.0 (educational project; https://github.com/caiiris/poem-classifier)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
DELAY_SEC = 1.0  # be nice to Wikipedia

# Plausible year range for "birth" or "active"
YEAR_MIN = 1600
YEAR_MAX = 2025


def extract_year_from_text(text: str) -> int | None:
    """Find a plausible birth/year in text. Prefer first 16xx-20xx."""
    if not text:
        return None
    # Match 4-digit years in reasonable range
    candidates = re.findall(r"\b(1[6-9]\d{2}|20[0-2]\d)\b", text)
    for y in candidates:
        year = int(y)
        if YEAR_MIN <= year <= YEAR_MAX:
            return year
    return None


def search_wikipedia(query: str) -> str | None:
    """Return best matching page title for query, or None."""
    r = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0].get("title")


def get_page_extract(title: str) -> str:
    """Get intro/extract text for a Wikipedia page."""
    r = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    return page.get("extract", "") or ""


def fetch_year_for_poet(poet: str) -> int | None:
    """Look up poet on Wikipedia and return a representative year (birth + 30), or None."""
    if not poet or poet == "nan":
        return None
    query = f"{poet} poet"
    title = search_wikipedia(query)
    if not title:
        return None
    time.sleep(DELAY_SEC)
    extract = get_page_extract(title)
    birth = extract_year_from_text(extract)
    if birth is None:
        return None
    # Birth + 30 as proxy for "typical publication period"
    year = birth + 30
    return min(year, YEAR_MAX)  # cap at current year


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch poet birth year + 30 from Wikipedia.")
    p.add_argument("--limit", type=int, default=None, help="Only poets from first N poems (for testing).")
    args = p.parse_args()

    print("Loading poems to get unique poets...")
    poems = pd.read_csv(POEMS_CSV)
    if args.limit:
        poems = poems.head(args.limit)
        print(f"Limited to first {args.limit} poems.")
    poets = poems["Poet"].astype(str).str.strip().unique()
    poets = [p for p in poets if p and p != "nan"]
    print(f"Found {len(poets)} unique poets.")

    # Resume: load existing cache so we don't re-fetch
    done: dict[str, int] = {}
    if CACHE_CSV.exists():
        cache = pd.read_csv(CACHE_CSV)
        for _, row in cache.iterrows():
            p = str(row["Poet"]).strip()
            y = pd.to_numeric(row["Year"], errors="coerce")
            if pd.notna(y):
                done[p] = int(y)
        print(f"Resuming: {len(done)} poets already have years.")

    rows = []
    for i, poet in enumerate(poets):
        if poet in done:
            rows.append({"Poet": poet, "Year": done[poet]})
            continue
        try:
            year = fetch_year_for_poet(poet)
            if year is not None:
                done[poet] = year
                rows.append({"Poet": poet, "Year": year})
            # Save progress periodically
            if (i + 1) % 50 == 0 and rows:
                pd.DataFrame(rows).to_csv(MAPPING_CSV, index=False)
                pd.DataFrame([{"Poet": k, "Year": v} for k, v in done.items()]).to_csv(
                    CACHE_CSV, index=False
                )
                print(f"  Progress: {len(done)} poets with years...")
        except Exception as e:
            print(f"  Skip {poet!r}: {e}")
        time.sleep(DELAY_SEC)

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["Poet"], keep="first")
    out.to_csv(MAPPING_CSV, index=False)
    if CACHE_CSV.exists():
        CACHE_CSV.unlink()
    print(f"Wrote {len(out)} poets to {MAPPING_CSV}")


if __name__ == "__main__":
    main()
