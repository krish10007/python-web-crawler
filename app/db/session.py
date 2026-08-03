import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.environ["DATABASE_URL"]

# Defaults are pool_size=5, max_overflow=10 (max 15 checked-out connections).
# Under Locust's 50 concurrent users that saturates quickly: checkout waits
# inflate p99 while the median query stays fast. Size the pool for ~50
# concurrent request handlers; stay well under Postgres' default
# max_connections=100 for a single API process.
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
