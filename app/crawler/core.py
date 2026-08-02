import asyncio
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from app.crawler.dedup import UrlDeduplicator
from app.crawler.fetcher import fetch


def extract_links(html: str, base_url: str) -> list[str]:
    """
    Parse all <a href="..."> links out of a page's HTML and resolve
    them to absolute URLs (a page might link to "/about", which only
    means something once joined with the page's own URL).
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        absolute = urljoin(base_url, tag["href"])
        # Strip URL fragments (#section) - they point at the same page,
        # not a new one, and would cause false "new URL" duplicates.
        absolute = absolute.split("#")[0]
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https"):
            links.append(absolute)
    return links


@dataclass
class CrawlStats:
    pages_crawled: int = 0
    pages_failed: int = 0
    urls_discovered: int = 0


async def worker(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    visited: UrlDeduplicator,
    stats: CrawlStats,
    stats_lock: asyncio.Lock,
    max_pages: int,
    max_depth: int,
) -> None:
    while True:
        url, depth = await queue.get()
        try:
            # Reserve a page slot atomically so concurrent workers cannot
            # both pass the limit check before either increments.
            async with stats_lock:
                if stats.pages_crawled >= max_pages:
                    # We've hit our limit - don't fetch, just drain the queue
                    # so other workers see it empty and the crawl can end.
                    continue
                stats.pages_crawled += 1
                current_count = stats.pages_crawled

            html = await fetch(session, url)
            if html is None:
                stats.pages_failed += 1
                continue

            print(f"[{name}] ({current_count}/{max_pages}) depth={depth} {url}")

            if depth < max_depth:
                for link in extract_links(html, url):
                    if visited.mark_seen(link):
                        stats.urls_discovered += 1
                        await queue.put((link, depth + 1))
        finally:
            queue.task_done()


async def crawl(seed_urls: list[str], num_workers: int = 10, max_pages: int = 50, max_depth: int = 3) -> CrawlStats:
    queue: asyncio.Queue = asyncio.Queue()
    visited = UrlDeduplicator()
    stats = CrawlStats()

    for url in seed_urls:
        visited.mark_seen(url)
        await queue.put((url, 0))

    stats_lock = asyncio.Lock()
    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(
                worker(f"worker-{i}", queue, session, visited, stats, stats_lock, max_pages, max_depth)
            )
            for i in range(num_workers)
        ]

        # Wait until every item put on the queue has been processed
        # (task_done() called for it) - this is how we detect "crawl finished"
        # without workers needing to coordinate with each other directly.
        await queue.join()

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    return stats


async def _demo() -> None:
    stats = await crawl(
        seed_urls=["https://example.com"],
        num_workers=5,
        max_pages=20,
        max_depth=2,
    )
    print(f"\nDone. Crawled={stats.pages_crawled} Failed={stats.pages_failed} Discovered={stats.urls_discovered}")


if __name__ == "__main__":
    asyncio.run(_demo())
