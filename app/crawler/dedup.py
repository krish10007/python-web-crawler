from pybloom_live import ScalableBloomFilter


class UrlDeduplicator:
    """
    Tracks which URLs have already been seen, using a bloom filter
    instead of a set() to keep memory flat as the crawl scales.

    Trade-off: false positives are possible (may occasionally think a
    genuinely-new URL was already seen, causing us to skip it) but false
    negatives are impossible (never re-crawls a URL it's truly seen).
    For a crawler, skipping a rare page is a negligible cost - re-crawling
    duplicates at scale is the real cost we're avoiding.
    """

    def __init__(self, initial_capacity: int = 100_000, error_rate: float = 0.001):
        # ScalableBloomFilter grows automatically as more items are added
        # past initial_capacity, instead of needing a hard-coded max size
        # up front - important since we don't know the final page count
        # before a crawl starts.
        self._filter = ScalableBloomFilter(
            initial_capacity=initial_capacity, error_rate=error_rate
        )

    def has_seen(self, url: str) -> bool:
        return url in self._filter

    def mark_seen(self, url: str) -> bool:
        """
        Adds the URL and returns True if it was newly added (i.e. was
        not already present). Returns False if it was already seen.
        This mirrors set.add() semantics you'd use with a plain set.
        """
        was_new = url not in self._filter
        self._filter.add(url)
        return was_new

    def __len__(self) -> int:
        return len(self._filter)
