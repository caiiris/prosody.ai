"""
Optimized scraper: fill missing Year from Poetry Foundation copyright lines.
- Groups poems by poet -> loads poet page once -> visits poem pages
- Runs 4 parallel browser tabs for speed
- Falls back to search if poet page doesn't have the poem link
- Resume-safe via cache file

Expected: ~1-2 hours for ~8000 missing poems (vs ~11 hours sequential).

Run after add_publication_year.py:
    pip install playwright && playwright install chromium
    python scripts/fill_missing_years_from_pf.py [--limit N]
"""
import asyncio
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WITH_YEAR_CSV = DATA_DIR / "PoetryFoundationData_with_year.csv"
CACHE_CSV = DATA_DIR / "pf_fill_missing_cache.csv"

MAX_YEAR = datetime.now().year
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20[0-2]\d)\b")
BASE_URL = "https://www.poetryfoundation.org"
DELAY_SEC = 0.75
NUM_WORKERS = 4


def normalize(s: str) -> str:
    return str(s).replace("\r", " ").replace("\n", " ").strip()


def poet_to_slug(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9\s-]", "", name.lower())
    name = re.sub(r"[\s-]+", "-", name).strip("-")
    return name


def extract_year(text: str) -> int | None:
    if not text:
        return None
    def valid(y):
        return 1600 <= y <= MAX_YEAR
    for pat in [
        r"[Pp]oem copyright\s*©\s*(\d{4})",
        r"copyright\s*©\s*(\d{4})",
        r"\([A-Za-z]+\s*,\s*(\d{4})\)",
        r"[Ff]irst published[^.]*?\b(1[6-9]\d{2}|20[0-2]\d)\b",
        r"[Pp]ublished[^.]*?\b(1[6-9]\d{2}|20[0-2]\d)\b",
    ]:
        m = re.search(pat, text)
        if m:
            y = int(m.group(1))
            if valid(y):
                return y
    parts = re.split(r"\s*(?:Notes|Copyright|Source)\s*:", text, maxsplit=1, flags=re.I)
    if len(parts) > 1:
        for cand in YEAR_RE.findall(parts[1][:1200]):
            y = int(cand)
            if valid(y):
                return y
    return None


def title_key(title: str) -> str:
    """Normalize title for fuzzy matching between CSV and page links."""
    t = normalize(title).lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


async def get_poem_links_from_poet_page(page, poet_name: str) -> dict[str, str]:
    """Load poet page, return {title_key: url} for all poem links found."""
    slug = poet_to_slug(poet_name)
    url = f"{BASE_URL}/poets/{slug}"
    try:
        await page.goto(url, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(0.4)
        links = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                .filter(a => /\\/poems?\\//.test(a.href) && !a.href.includes('search'))
                .map(a => ({href: a.href, text: a.textContent.trim()}))
        """)
        result = {}
        for link in (links or []):
            href = link.get("href", "")
            text = link.get("text", "")
            if href and text:
                result[title_key(text)] = href
        return result
    except Exception:
        return {}


async def get_year_from_poem_page(page, url: str) -> int | None:
    try:
        await page.goto(url, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(0.3)
        text = await page.evaluate("() => document.body.innerText") or ""
        return extract_year(text)
    except Exception:
        return None


async def search_poem_url(page, title: str, poet: str) -> str | None:
    """Fallback: search PF for a specific poem."""
    from bs4 import BeautifulSoup
    query = f"{normalize(title)} {normalize(poet)}".strip()[:120]
    url = f"{BASE_URL}/search?query={quote_plus(query)}&refinement=poems"
    try:
        await page.goto(url, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(0.4)
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/poems/" in href or "/poem/" in href:
                full = urljoin(BASE_URL, href)
                if "search" not in full:
                    return full
    except Exception:
        pass
    return None


async def process_poet(poet_name, poems_for_poet, cache, semaphore, context):
    """Process all missing poems for one poet. Returns {df_index: year}."""
    results = {}
    async with semaphore:
        page = await context.new_page()
        try:
            # Step 1: load poet page to get poem links
            await asyncio.sleep(DELAY_SEC)
            poem_links = await get_poem_links_from_poet_page(page, poet_name)

            for idx, raw_title in poems_for_poet:
                key = (normalize(raw_title), poet_name)
                if key in cache:
                    results[idx] = cache[key]
                    continue

                # Step 2: match title to a link from poet page
                tk = title_key(raw_title)
                url = poem_links.get(tk)

                # Step 3: fallback to search if no match
                if not url:
                    await asyncio.sleep(DELAY_SEC)
                    url = await search_poem_url(page, raw_title, poet_name)

                if not url:
                    continue

                # Step 4: visit poem page and extract year
                await asyncio.sleep(DELAY_SEC)
                year = await get_year_from_poem_page(page, url)
                if year is not None:
                    cache[key] = year
                    results[idx] = year
        except Exception:
            pass
        finally:
            await page.close()
    return results


def save_cache(cache: dict):
    if cache:
        pd.DataFrame([
            {"Title": k[0], "Poet": k[1], "Year": v} for k, v in cache.items()
        ]).to_csv(CACHE_CSV, index=False)


async def main_async(limit=None):
    from playwright.async_api import async_playwright

    if not WITH_YEAR_CSV.exists():
        print(f"Not found: {WITH_YEAR_CSV}. Run add_publication_year.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(WITH_YEAR_CSV)
    df["Title"] = df["Title"].astype(str)
    df["Poet"] = df["Poet"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    missing = df[df["Year"].isna()]
    print(f"Poems missing Year: {len(missing)}")

    # Group by poet
    poet_groups: dict[str, list] = {}
    for idx, row in missing.iterrows():
        poet = row["Poet"]
        if not poet or poet == "nan":
            continue
        poet_groups.setdefault(poet, []).append((idx, row["Title"]))
    print(f"Unique poets to process: {len(poet_groups)}")

    if limit:
        poets_list = list(poet_groups.items())[:limit]
        poet_groups = dict(poets_list)
        total_poems = sum(len(v) for v in poet_groups.values())
        print(f"Limited to first {limit} poets ({total_poems} poems).")

    # Load cache
    cache: dict[tuple[str, str], int] = {}
    if CACHE_CSV.exists():
        try:
            c = pd.read_csv(CACHE_CSV)
            for _, r in c.iterrows():
                k = (normalize(r["Title"]), str(r["Poet"]).strip())
                y = pd.to_numeric(r["Year"], errors="coerce")
                if pd.notna(y):
                    cache[k] = int(y)
            print(f"Resuming: {len(cache)} already cached.")
        except Exception:
            pass

    semaphore = asyncio.Semaphore(NUM_WORKERS)
    all_results = {}
    poets_list = list(poet_groups.items())

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="PoetryEraProject/1.0 (educational)"
        )

        # Process in batches of 50 poets at a time, save progress after each batch
        batch_size = 50
        for i in range(0, len(poets_list), batch_size):
            batch = poets_list[i : i + batch_size]
            tasks = [
                process_poet(name, poems, cache, semaphore, context)
                for name, poems in batch
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results_list:
                if isinstance(r, dict):
                    all_results.update(r)
            save_cache(cache)
            done = min(i + batch_size, len(poets_list))
            print(f"  {done}/{len(poets_list)} poets done, {len(all_results)} poems filled so far...")

        await browser.close()

    save_cache(cache)

    # Merge results back into dataframe (keep ALL poems, just fill Year where found)
    for idx, year in all_results.items():
        df.at[idx, "Year"] = year

    has_year = int(df["Year"].notna().sum())
    still_missing = int(df["Year"].isna().sum())
    df.to_csv(WITH_YEAR_CSV, index=False)
    print(f"Done! Wrote {len(df)} poems to {WITH_YEAR_CSV}")
    print(f"  {has_year} with Year, {still_missing} still missing.")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Fill missing Year from PF copyright lines (optimized)."
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N poets (for testing).",
    )
    args = p.parse_args()
    asyncio.run(main_async(limit=args.limit))


if __name__ == "__main__":
    main()
