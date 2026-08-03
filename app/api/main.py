import time
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Page, Posting, Term
from app.db.session import get_session
from app.index.text_processing import tokenize

app = FastAPI(title="Web Crawler Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
async def search(
    q: Annotated[str, Query(min_length=1, description="Search query")],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Rank pages by summed TF-IDF across all stemmed query terms that
    appear in the inverted index.
    """
    query_terms = tokenize(q)
    if not query_terms:
        return {"results": [], "query_time_ms": 0.0}

    stmt = (
        select(
            Page.url,
            Page.title,
            func.sum(Posting.tfidf_score).label("score"),
        )
        .join(Posting, Posting.page_id == Page.id)
        .join(Term, Term.id == Posting.term_id)
        .where(Term.term.in_(query_terms))
        .group_by(Page.id, Page.url, Page.title)
        .order_by(func.sum(Posting.tfidf_score).desc())
        .limit(limit)
    )

    started = time.perf_counter()
    rows = (await session.execute(stmt)).all()
    query_time_ms = (time.perf_counter() - started) * 1000.0

    return {
        "results": [
            {
                "url": row.url,
                "title": row.title,
                "score": float(row.score) if row.score is not None else 0.0,
            }
            for row in rows
        ],
        "query_time_ms": round(query_time_ms, 3),
    }
