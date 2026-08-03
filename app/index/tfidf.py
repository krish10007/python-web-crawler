import asyncio
import math
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Page, Posting, Term
from app.index.text_processing import extract_text, tokenize


def compute_term_frequencies(tokens: list[str]) -> dict[str, int]:
    """Count how many times each token appears in the token list."""
    return dict(Counter(tokens))


async def _get_or_create_term(session: AsyncSession, term_str: str) -> Term:
    """Insert term if missing; safe under concurrent workers via ON CONFLICT."""
    await session.execute(
        pg_insert(Term)
        .values(term=term_str, document_frequency=0)
        .on_conflict_do_nothing(index_elements=["term"])
    )
    return (
        await session.execute(select(Term).where(Term.term == term_str))
    ).scalar_one()


async def index_page(session: AsyncSession, url: str, html: str) -> None:
    """
    Parse one page's HTML and write/update its rows in the inverted index.

    Creates (or updates) a Page, get-or-creates a Term for each unique
    stemmed token, and writes a Posting per (term, page) with raw
    term_frequency. tfidf_score is left at 0.0 — real scores need corpus-wide
    document frequencies, which only exist after the full crawl is indexed
    (see recompute_tfidf_scores).
    """
    title, body_text = extract_text(html)
    # Index title words too — they are usually highly relevant to the page.
    tokens = tokenize(f"{title} {body_text}")
    term_freqs = compute_term_frequencies(tokens)
    now = datetime.now(timezone.utc)

    # Retry on transient Postgres deadlocks from concurrent term inserts.
    for attempt in range(5):
        try:
            result = await session.execute(select(Page).where(Page.url == url))
            page = result.scalar_one_or_none()

            if page is None:
                page = Page(
                    url=url,
                    title=title[:512] if title else None,
                    word_count=len(tokens),
                    crawled_at=now,
                )
                session.add(page)
                await session.flush()
            else:
                page.title = title[:512] if title else None
                page.word_count = len(tokens)
                page.crawled_at = now
                # Replacing an existing page's index: drop old postings first so
                # the (term_id, page_id) unique constraint isn't violated.
                await session.execute(delete(Posting).where(Posting.page_id == page.id))
                await session.flush()

            # Sorted order keeps lock acquisition consistent across workers,
            # which sharply reduces deadlock rate under concurrency.
            for term_str, count in sorted(term_freqs.items()):
                term = await _get_or_create_term(session, term_str[:128])
                session.add(
                    Posting(
                        term_id=term.id,
                        page_id=page.id,
                        term_frequency=count,
                        tfidf_score=0.0,
                    )
                )

            await session.commit()
            return
        except (DBAPIError, OperationalError) as exc:
            await session.rollback()
            if attempt == 4 or "deadlock" not in str(exc).lower():
                raise
            await asyncio.sleep(0.05 * (attempt + 1))


async def recompute_tfidf_scores(session: AsyncSession) -> None:
    """
    Refresh document_frequency on every Term and tfidf_score on every Posting.

    Run this once after a crawl has finished indexing all pages. IDF is a
    corpus-level statistic — it only makes sense once every page is known.

    TF-IDF in plain English
    -----------------------
    TF-IDF answers: "How characteristic is this word of this page?"

    It multiplies two ideas:

    1. Term Frequency (TF)
       How often the term appears *in this page*, divided by the page's
       total word count so longer pages don't automatically look more
       relevant just because they have more words:
           tf = term_frequency / word_count

    2. Inverse Document Frequency (IDF)
       How rare the term is *across the whole corpus*. A word that appears
       on almost every page ("website", "page") is a weak relevance signal;
       a word that appears on only a few pages is a strong one. We use the
       smoothed, non-negative form:
           idf = max(0, ln(N / (1 + df)))
       where N is the total number of pages and df (document_frequency) is
       how many distinct pages contain the term. The "+1" in the denominator
       is Laplace smoothing — it avoids division by zero if df is somehow 0
       and gently dampens scores for very rare terms.

       The max(0, ...) clamp matters for small corpora: when a term appears
       in nearly every document, ln(N / (1 + df)) goes negative. Without
       clamping, those common terms would *subtract* from a page's score
       instead of simply contributing nothing. Flooring at 0 means
       ubiquitous terms are ignored rather than actively hurting relevance.

    Final score:
           tfidf_score = tf * idf

    A high score means: the term shows up a lot on this page, and not on
    many other pages — i.e. it's a good fingerprint for this document.
    """
    total_page_count = await session.scalar(select(func.count()).select_from(Page))
    if not total_page_count:
        return

    # document_frequency = number of distinct pages that have a posting
    # for this term. Because of uq_term_page, that's just COUNT(postings).
    df_rows = await session.execute(
        select(Posting.term_id, func.count(Posting.page_id)).group_by(Posting.term_id)
    )
    df_map = dict(df_rows.all())

    terms = (await session.execute(select(Term))).scalars().all()
    for term in terms:
        term.document_frequency = df_map.get(term.id, 0)

    postings = (
        await session.execute(
            select(Posting).options(
                selectinload(Posting.page),
                selectinload(Posting.term),
            )
        )
    ).scalars().all()

    for posting in postings:
        word_count = posting.page.word_count
        if word_count <= 0:
            posting.tfidf_score = 0.0
            continue

        tf = posting.term_frequency / word_count
        idf = max(
            0.0,
            math.log(total_page_count / (1 + posting.term.document_frequency)),
        )
        posting.tfidf_score = tf * idf

    await session.commit()
