"""Stage 1 — Harvest Wikipedia articles via the Wikimedia REST API."""

import json
import re
import time

import requests
from tqdm import tqdm

from .config import RAW_ARTICLES_DIR, SEED_ARTICLES

API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_DELAY = 0.5  # seconds between requests


def sanitize_title(title: str) -> str:
    """Convert article title to a safe filename."""
    return re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()


def fetch_article(title: str, session: requests.Session) -> dict | None:
    """Fetch a single Wikipedia article. Returns dict with title + text, or None."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "redirects": 1,  # follow redirects
    }
    resp = session.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            print(f"  WARNING: Article not found: '{title}'")
            return None
        canonical_title = page.get("title", title)
        extract = page.get("extract", "")
        if not extract:
            print(f"  WARNING: Empty extract for '{title}'")
            return None
        return {
            "title": canonical_title,
            "original_query": title,
            "text": extract,
            "length_chars": len(extract),
        }
    return None


def run():
    """Harvest all seed articles and save to disk."""
    print("=" * 60)
    print("STAGE 1 — Harvesting Wikipedia articles")
    print("=" * 60)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "LITMDatasetBuilder/1.0 (research project; Python/requests)"
    })

    metadata = []
    fetched = 0
    skipped = 0

    for title in tqdm(SEED_ARTICLES, desc="Fetching articles"):
        article = fetch_article(title, session)
        if article is None:
            skipped += 1
            continue

        # Save article text
        filename = sanitize_title(article["title"]) + ".txt"
        filepath = RAW_ARTICLES_DIR / filename
        filepath.write_text(article["text"], encoding="utf-8")

        metadata.append({
            "title": article["title"],
            "original_query": article["original_query"],
            "filename": filename,
            "length_chars": article["length_chars"],
        })
        fetched += 1
        time.sleep(REQUEST_DELAY)

    # Save metadata
    meta_path = RAW_ARTICLES_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone: {fetched} articles fetched, {skipped} skipped.")
    print(f"Saved to {RAW_ARTICLES_DIR}")
    return metadata


if __name__ == "__main__":
    run()
