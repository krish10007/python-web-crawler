from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Page, Posting
from app.index.tfidf import compute_term_frequencies, index_page, recompute_tfidf_scores


def test_compute_term_frequencies_counts_known_list():
    tokens = ["alpha", "beta", "alpha", "gamma", "alpha", "beta"]
    assert compute_term_frequencies(tokens) == {
        "alpha": 3,
        "beta": 2,
        "gamma": 1,
    }


async def test_recompute_tfidf_rare_term_outranks_common_term(
    db_session: AsyncSession,
):
    """
    Core TF-IDF property: a term unique to one page should score higher
    on that page than a term that appears on every page.
    """
    pages = [
        (
            "https://test.example/rare",
            "<html><title>Doc One</title><body>rareword commonword commonword</body></html>",
        ),
        (
            "https://test.example/mid",
            "<html><title>Doc Two</title><body>commonword otherword</body></html>",
        ),
        (
            "https://test.example/common",
            "<html><title>Doc Three</title><body>commonword fillerword</body></html>",
        ),
    ]
    for url, html in pages:
        await index_page(db_session, url, html)

    await recompute_tfidf_scores(db_session)

    result = await db_session.execute(
        select(Page)
        .where(Page.url == "https://test.example/rare")
        .options(selectinload(Page.postings).selectinload(Posting.term))
    )
    rare_page = result.scalar_one()
    scores = {p.term.term: p.tfidf_score for p in rare_page.postings}

    assert "rareword" in scores
    assert "commonword" in scores
    assert scores["rareword"] > scores["commonword"]

    # Regression: clamped IDF must never produce negative scores.
    all_postings = (
        await db_session.execute(select(Posting))
    ).scalars().all()
    assert all_postings, "expected postings after indexing"
    assert all(p.tfidf_score >= 0.0 for p in all_postings)
