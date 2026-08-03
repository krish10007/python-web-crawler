"""
Shared fixtures: session-scoped crawler_db_test database, per-test sessions.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from urllib.parse import urlparse, urlunparse

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# venv lives inside the project cwd; NLTK's import guard blocks `regex` otherwise.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

load_dotenv()

from app.db.models import Base  # noqa: E402

TEST_DB_NAME = "crawler_db_test"


def _swap_db_name(database_url: str, db_name: str) -> str:
    """Replace the path database name in a SQLAlchemy URL."""
    parsed = urlparse(database_url)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


@pytest_asyncio.fixture(scope="session")
async def test_engine() -> AsyncIterator[AsyncEngine]:
    """
    Create crawler_db_test, apply schema, yield an engine, then drop the DB.
    Never touches the real crawler_db used by crawls.
    """
    base_url = os.environ["DATABASE_URL"]
    test_url = _swap_db_name(base_url, TEST_DB_NAME)
    admin_url = _swap_db_name(base_url, "postgres")

    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with admin_engine.connect() as conn:
        await conn.execute(
            text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        )
        await conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    await admin_engine.dispose()

    engine = create_async_engine(test_url, echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()

    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with admin_engine.connect() as conn:
        await conn.execute(
            text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        )
    await admin_engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Function-scoped session; truncates tables after each test."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE postings, pages, terms RESTART IDENTITY CASCADE")
        )
