import uuid

from app.crawler.dedup import UrlDeduplicator


def test_mark_seen_returns_true_for_new_url():
    dedup = UrlDeduplicator()
    assert dedup.mark_seen("https://example.com/a") is True


def test_mark_seen_returns_false_for_duplicate():
    dedup = UrlDeduplicator()
    url = "https://example.com/page"
    assert dedup.mark_seen(url) is True
    assert dedup.mark_seen(url) is False


def test_len_reflects_unique_urls_added():
    dedup = UrlDeduplicator()
    assert len(dedup) == 0
    dedup.mark_seen("https://example.com/1")
    dedup.mark_seen("https://example.com/2")
    dedup.mark_seen("https://example.com/1")  # duplicate
    assert len(dedup) == 2


def test_false_positive_rate_stays_near_configured_error_rate():
    """
    Bloom filters allow rare false positives. After inserting N known URLs,
    probe many never-inserted URLs and assert the observed FP rate stays
    under roughly 2x the configured error_rate — not zero.
    """
    error_rate = 0.01
    dedup = UrlDeduplicator(initial_capacity=2_000, error_rate=error_rate)

    known = [f"https://known.example/{i}" for i in range(1_000)]
    for url in known:
        dedup.mark_seen(url)

    # Definitely-different URLs (UUID paths) that were never inserted.
    probes = 5_000
    false_positives = 0
    for _ in range(probes):
        novel = f"https://novel.example/{uuid.uuid4()}"
        if dedup.has_seen(novel):
            false_positives += 1

    observed_rate = false_positives / probes
    assert observed_rate < error_rate * 2, (
        f"false positive rate {observed_rate:.4f} exceeded 2x error_rate "
        f"({error_rate * 2:.4f}); got {false_positives}/{probes} FPs"
    )
