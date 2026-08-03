"""
Crawl throughput benchmark: pages/minute at 10, 25, and 50 workers.

Seed strategy
-------------
Wikipedia was the original target, but Wikimedia CDN returns HTTP 403 for
this crawler ("Please respect our robot policy...") even with a descriptive
User-Agent. So this benchmark uses *true multi-domain* seeds instead.

That is actually the right setup for measuring worker scaling: the Redis
rate limiter caps ~1 req/s *per domain*, so multiple domains let workers
proceed in parallel. A single-domain Wikipedia run would have shown the
same pages/min at 10, 25, and 50 workers.
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

from app.crawler.core import crawl

# True multi-domain set (one host per seed) so per-domain rate limiting
# does not serialize the entire crawl onto a single 1 req/s budget.
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

WORKER_COUNTS = [10, 25, 50]
MAX_PAGES = 300
MAX_DEPTH = 3


async def run_one(num_workers: int) -> dict:
    # Each crawl() call builds a fresh UrlDeduplicator, queue, and stats.
    print(f"\n=== workers={num_workers} max_pages={MAX_PAGES} max_depth={MAX_DEPTH} ===")
    started = time.perf_counter()
    stats = await crawl(
        seed_urls=SEED_URLS,
        num_workers=num_workers,
        max_pages=MAX_PAGES,
        max_depth=MAX_DEPTH,
    )
    elapsed = time.perf_counter() - started
    pages_per_minute = (stats.pages_crawled / elapsed) * 60 if elapsed > 0 else 0.0

    print(
        f"workers={num_workers}  pages_crawled={stats.pages_crawled}  "
        f"elapsed_sec={elapsed:.2f}  pages_per_min={pages_per_minute:.2f}  "
        f"failed={stats.pages_failed}  discovered={stats.urls_discovered}"
    )
    return {
        "worker_count": num_workers,
        "pages_crawled": stats.pages_crawled,
        "elapsed_sec": elapsed,
        "pages_per_min": pages_per_minute,
    }


async def main() -> None:
    print("Crawl throughput benchmark")
    print("=" * 72)
    print(
        "NOTE: Wikipedia seeds were abandoned — Wikimedia returns HTTP 403\n"
        "for this bot traffic. Using true multi-domain seeds instead so\n"
        "per-domain rate limiting (~1 req/s each) can run in parallel and\n"
        "worker concurrency is actually measurable."
    )
    domains = sorted({__import__("urllib.parse", fromlist=["urlparse"]).urlparse(u).netloc for u in SEED_URLS})
    print(f"Domains ({len(domains)}): {', '.join(domains)}")
    print(f"Seeds ({len(SEED_URLS)}):")
    for url in SEED_URLS:
        print(f"  - {url}")

    results: list[dict] = []
    for n in WORKER_COUNTS:
        results.append(await run_one(n))

    print("\n" + "=" * 72)
    print("RESULTS TABLE (paste into README)")
    print("=" * 72)
    print(
        f"| {'worker_count':>12} | {'pages_crawled':>13} | "
        f"{'elapsed_sec':>11} | {'pages_per_min':>13} |"
    )
    print(
        f"| {'---':>12} | {'---':>13} | {'---':>11} | {'---':>13} |"
    )
    for r in results:
        print(
            f"| {r['worker_count']:>12} | {r['pages_crawled']:>13} | "
            f"{r['elapsed_sec']:>11.2f} | {r['pages_per_min']:>13.2f} |"
        )
    print()
    print(
        f"Multi-domain crawl across {len(domains)} hosts; ~1 req/s per domain.\n"
        "Higher worker counts should improve pages/min while the queue stays\n"
        "populated across many domains (until another bottleneck dominates)."
    )


if __name__ == "__main__":
    asyncio.run(main())
