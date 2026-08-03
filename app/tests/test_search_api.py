from collections.abc import AsyncIterator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.api.main import app
from app.db.models import Page, Posting, Term
from app.db.session import get_session


@pytest_asyncio.fixture
async def client(test_engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    """HTTPX client with get_session overridden to use crawler_db_test."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()

    # Clean any rows the API tests wrote into the shared test DB.
    async with test_engine.begin() as conn:
        from sqlalchemy import text

        await conn.execute(
            text("TRUNCATE postings, pages, terms RESTART IDENTITY CASCADE")
        )


async def test_health_returns_ok(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_search_shape_and_score_ordering(
    client: AsyncClient, db_session: AsyncSession
):
    """
    Seed three pages with known TF-IDF postings for the same stemmed term,
    then assert /search returns the expected JSON shape and descending scores.
    """
    term = Term(term="widget", document_frequency=3)
    db_session.add(term)
    await db_session.flush()

    pages_and_scores = [
        ("https://test.example/high", "High Widget", 0.9),
        ("https://test.example/mid", "Mid Widget", 0.4),
        ("https://test.example/low", "Low Widget", 0.1),
    ]
    for url, title, score in pages_and_scores:
        page = Page(url=url, title=title, word_count=10)
        db_session.add(page)
        await db_session.flush()
        db_session.add(
            Posting(
                term_id=term.id,
                page_id=page.id,
                term_frequency=1,
                tfidf_score=score,
            )
        )
    await db_session.commit()

    response = await client.get("/search", params={"q": "widget", "limit": 10})
    assert response.status_code == 200
    body = response.json()

    assert "results" in body
    assert "query_time_ms" in body
    assert isinstance(body["query_time_ms"], (int, float))

    results = body["results"]
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["url"] == "https://test.example/high"
    assert results[-1]["url"] == "https://test.example/low"
    for item in results:
        assert set(item.keys()) >= {"url", "title", "score"}
