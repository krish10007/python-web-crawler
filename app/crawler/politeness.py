import asyncio
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import redis.asyncio as redis


def domain_from_url(url: str) -> str:
    """Extract the network location (host[:port]) from a URL."""
    return urlparse(url).netloc


class RobotsChecker:
    """
    Fetches and caches robots.txt per domain, then answers
    can-we-crawl-this-URL queries. Fail-open: if robots.txt is
    missing or unreadable, crawling is allowed.
    """

    def __init__(self) -> None:
        self._cache: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def _get_parser(self, url: str) -> RobotFileParser | None:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain in self._cache:
            return self._cache[domain]

        async with self._lock_for(domain):
            if domain in self._cache:
                return self._cache[domain]

            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                # RobotFileParser.read() is blocking I/O - offload it.
                await asyncio.to_thread(parser.read)
            except Exception:
                # Fail open: cache None so we don't retry every URL.
                self._cache[domain] = None
                return None

            self._cache[domain] = parser
            return parser

    async def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        parser = await self._get_parser(url)
        if parser is None:
            return True
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:
            return True


class RateLimiter:
    """
    Per-domain crawl delay backed by Redis so multiple crawler
    processes can share the same politeness budget.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        # Serialize waits for the same domain within this process so
        # concurrent workers don't all read the same timestamp and burst.
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def wait_if_needed(
        self, domain: str, min_delay_seconds: float = 1.0
    ) -> None:
        key = f"ratelimit:{domain}"
        async with self._lock_for(domain):
            now = time.time()
            last_raw = await self._redis.get(key)
            if last_raw is not None:
                elapsed = now - float(last_raw)
                remaining = min_delay_seconds - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    now = time.time()
            await self._redis.set(key, str(now))
