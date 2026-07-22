from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    crawled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    postings: Mapped[list["Posting"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Page id={self.id} url={self.url!r}>"


class Term(Base):
    __tablename__ = "terms"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    term: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    # Number of distinct pages containing this term.
    # Stored directly so IDF can be computed without a live COUNT() at search time.
    document_frequency: Mapped[int] = mapped_column(Integer, default=0)

    postings: Mapped[list["Posting"]] = relationship(
        back_populates="term", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Term id={self.id} term={self.term!r}>"


class Posting(Base):
    """
    The inverted index itself: one row = one (term, page) pair,
    carrying the precomputed relevance signal for that pair.
    """

    __tablename__ = "postings"
    __table_args__ = (
        UniqueConstraint("term_id", "page_id", name="uq_term_page"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    term_id: Mapped[int] = mapped_column(ForeignKey("terms.id"), index=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"), index=True)
    term_frequency: Mapped[int] = mapped_column(Integer)
    tfidf_score: Mapped[float] = mapped_column()

    term: Mapped["Term"] = relationship(back_populates="postings")
    page: Mapped["Page"] = relationship(back_populates="postings")

    def __repr__(self) -> str:
        return f"<Posting term_id={self.term_id} page_id={self.page_id} score={self.tfidf_score:.4f}>"
