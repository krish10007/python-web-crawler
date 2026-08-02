import asyncio

import aiohttp


async def fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    """
    Fetch a single URL's HTML. Returns None on any failure instead of
    raising — a crawler must be resilient to individual page failures
    (timeouts, 404s, broken servers) without crashing the whole run.
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
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
