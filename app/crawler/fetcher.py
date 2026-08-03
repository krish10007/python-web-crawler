import asyncio

import aiohttp

# Many sites (notably Wikipedia) reject requests with no User-Agent.
DEFAULT_USER_AGENT = (
    "AcademicWebCrawler/1.0 (+local educational project; polite crawl)"
)
DEFAULT_HEADERS = {"User-Agent": DEFAULT_USER_AGENT}


async def fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    """
    Fetch a single URL's HTML. Returns None on any failure instead of
    raising — a crawler must be resilient to individual page failures
    (timeouts, 404s, broken servers) without crashing the whole run.
    """
    try:
        async with session.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status != 200:
                return None
            return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def _demo() -> None:
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, "https://example.com")
        print(f"Fetched {len(html) if html else 0} characters")


if __name__ == "__main__":
    asyncio.run(_demo())
