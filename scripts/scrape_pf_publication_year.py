"""
Scrape Poetry Foundation for each poem's publication year.
Use --limit N to process only the first N poems (for testing).

Reads PoetryFoundationData.csv, finds each poem on PF (via search), extracts
publication year from the poem page (e.g. "First published in X, YYYY"), and
writes PoetryFoundationData_with_year.csv: original columns + PublicationYear;
rows without a year are dropped.

Poetry Foundation's search and poem pages are JS-rendered, so this script uses
Playwright. Install with: pip install playwright && playwright install chromium
"""
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd

from datetime import datetime
MAX_YEAR = datetime.now().year  # ignore future years (e.g. from page footer)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POEMS_CSV = DATA_DIR / "PoetryFoundationData.csv"
OUTPUT_CSV = DATA_DIR / "PoetryFoundationData_with_year.csv"
CACHE_CSV = DATA_DIR / "scrape_publication_year_cache.csv"  # (Title, Poet) -> Year for resume

BASE_URL = "https://www.poetryfoundation.org"
SEARCH_URL = BASE_URL + "/search"
DELAY_SEC = 1.5  # be polite to the server
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20[0-2]\d)\b")


def extract_publication_year_from_text(text: str) -> int | None:
    """Parse page text for publication year. PF pages have it in Copyright Credit / (Month, YYYY)."""
    if not text:
        return None

    def valid(y: int) -> bool:
        return 1600 <= y <= MAX_YEAR

    # Priority 1: Poem copyright line, e.g. "Poem copyright ©2015 by ..." or "(February, 2015)"
    for pat in [
        r"[Pp]oem copyright\s*©\s*(\d{4})",
        r"copyright\s*©\s*(\d{4})",
        r"\([A-Za-z]+\s*,\s*(\d{4})\)",  # (February, 2015)
        r"[Ff]irst published[^.]*?\b(1[6-9]\d{2}|20[0-2]\d)\b",
        r"[Pp]ublished[^.]*?\b(1[6-9]\d{2}|20[0-2]\d)\b",
    ]:
        m = re.search(pat, text)
        if m:
            y = int(m.group(1))
            if valid(y):
                return y

    # Priority 2: Notes/Copyright/Source block (avoid page footer)
    parts = re.split(r"\s*(?:Notes|Copyright|Source)\s*:", text, maxsplit=1, flags=re.I)
    if len(parts) > 1:
        block = parts[1][:1200]  # first chunk of credit text
        for cand in YEAR_RE.findall(block):
            y = int(cand)
            if valid(y):
                return y

    # Fallback: any plausible year in the middle of the page (skip last ~800 chars to avoid footer)
    body = text[:-800] if len(text) > 1000 else text
    for cand in YEAR_RE.findall(body):
        y = int(cand)
        if valid(y):
            return y
    return None


def search_poem_url_playwright(page, title: str, poet: str) -> str | None:
    """Use Playwright to search PF and return first poem page URL."""
    query = f"{title} {poet}".strip()
    if not query:
        return None
    url = f"{SEARCH_URL}?query={quote_plus(query)}&refinement=poems"
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(1.0)  # allow dynamic content
        content = page.content()
    except Exception:
        return None
    # Parse for links to poem pages
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/poems/" in href or "/poem/" in href:
            full = urljoin(BASE_URL, href)
            if "search" not in full:
                return full
    return None


def fetch_poem_year_playwright(page, url: str) -> int | None:
    """Use Playwright to load poem page and extract publication year from visible text."""
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(0.8)
        text = page.evaluate("() => document.body.innerText") or ""
        return extract_publication_year_from_text(text)
    except Exception:
        return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="Scrape PF for per-poem publication year.")
    p.add_argument("--limit", type=int, default=None, help="Process only first N poems (for testing).")
    args = p.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is required (PF site is JS-rendered). Install with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return

    print("Loading poems...")
    poems = pd.read_csv(POEMS_CSV)
    poems["Title"] = poems["Title"].astype(str).str.strip()
    poems["Poet"] = poems["Poet"].astype(str).str.strip()
    if args.limit:
        poems = poems.head(args.limit)
        print(f"Limited to first {args.limit} poems.")

    # Resume: load cache of (Title, Poet) -> Year
    cache: dict[tuple[str, str], int] = {}
    if CACHE_CSV.exists():
        try:
            df = pd.read_csv(CACHE_CSV)
            for _, row in df.iterrows():
                key = (str(row["Title"]).strip(), str(row["Poet"]).strip())
                y = pd.to_numeric(row["PublicationYear"], errors="coerce")
                if pd.notna(y):
                    cache[key] = int(y)
            print(f"Resuming: {len(cache)} poems already have publication year.")
        except Exception:
            pass

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(user_agent="PoetryEraProject/1.0 (educational)")
        page = context.new_page()
        try:
            for i, row in poems.iterrows():
                title, poet = row["Title"], row["Poet"]
                if not title or not poet or poet == "nan":
                    continue
                key = (title, poet)
                if key in cache:
                    results.append((i, cache[key]))
                    continue
                time.sleep(DELAY_SEC)
                url = search_poem_url_playwright(page, title, poet)
                if not url:
                    continue
                time.sleep(DELAY_SEC)
                year = fetch_poem_year_playwright(page, url)
                if year is not None:
                    cache[key] = year
                    results.append((i, year))
                if len(results) % 100 == 0 and results:
                    print(f"  Progress: {len(results)} poems with year...")
                    pd.DataFrame(
                        [{"Title": k[0], "Poet": k[1], "PublicationYear": v} for k, v in cache.items()]
                    ).to_csv(CACHE_CSV, index=False)
        finally:
            browser.close()

    # Save cache for resume
    if cache:
        pd.DataFrame(
            [{"Title": k[0], "Poet": k[1], "PublicationYear": v} for k, v in cache.items()]
        ).to_csv(CACHE_CSV, index=False)

    # Merge back: add PublicationYear column, keep only rows with year
    year_by_idx = dict(results)
    poems["PublicationYear"] = poems.index.map(year_by_idx)
    out = poems.dropna(subset=["PublicationYear"])
    out["PublicationYear"] = out["PublicationYear"].astype(int)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} poems to {OUTPUT_CSV} (dropped {len(poems) - len(out)} without year).")


if __name__ == "__main__":
    main()
