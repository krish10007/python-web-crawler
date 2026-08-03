"""
Single full crawl: max_pages=1000, max_depth=3, num_workers=50.

Reuses the proven multi-domain seed list from scripts/benchmark_crawl.py
(Wikipedia abandoned — Wikimedia 403s this bot traffic).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text

from app.crawler.core import crawl
from app.db.session import AsyncSessionLocal

# Same true multi-domain set as scripts/benchmark_crawl.py
SEED_URLS = [
    "https://example.com",
    "https://www.iana.org/domains/example",
    "https://www.python.org/",
    "https://docs.python.org/3/",
    "https://www.w3.org/",
    "https://quotes.toscrape.com/",
    "https://books.toscrape.com/",
    "https://httpbin.org/html",
    "https://info.cern.ch/",
    "https://www.rfc-editor.org/",
]

NUM_WORKERS = 50
MAX_PAGES = 1000
MAX_DEPTH = 3


async def table_counts() -> dict[str, int]:
    async with AsyncSessionLocal() as session:
        counts: dict[str, int] = {}
        for table in ("pages", "terms", "postings"):
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[table] = int(result.scalar_one())
        return counts


async def main() -> None:
    domains = sorted(
        {
            __import__("urllib.parse", fromlist=["urlparse"]).urlparse(u).netloc
            for u in SEED_URLS
        }
    )
    print("Full crawl")
    print("=" * 72)
    print(f"workers={NUM_WORKERS}  max_pages={MAX_PAGES}  max_depth={MAX_DEPTH}")
    print(f"Domains ({len(domains)}): {', '.join(domains)}")
    print(f"Seeds ({len(SEED_URLS)}):")
    for url in SEED_URLS:
        print(f"  - {url}")
    print()

    started = time.perf_counter()
    stats = await crawl(
        seed_urls=SEED_URLS,
        num_workers=NUM_WORKERS,
        max_pages=MAX_PAGES,
        max_depth=MAX_DEPTH,
    )
    elapsed = time.perf_counter() - started
    pages_per_minute = (stats.pages_crawled / elapsed) * 60 if elapsed > 0 else 0.0

    print("\n" + "=" * 72)
    print("CRAWL RESULTS")
    print("=" * 72)
    print(f"pages_crawled : {stats.pages_crawled}")
    print(f"elapsed_sec   : {elapsed:.2f}")
    print(f"pages_per_min : {pages_per_minute:.2f}")
    print(f"failed        : {stats.pages_failed}")
    print(f"discovered    : {stats.urls_discovered}")

    counts = await table_counts()
    print("\nDB COUNTS")
    print("=" * 72)
    for table, count in counts.items():
        print(f"SELECT COUNT(*) FROM {table}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
