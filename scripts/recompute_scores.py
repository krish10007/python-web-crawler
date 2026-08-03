"""
Recompute TF-IDF scores for the existing inverted index (no re-crawl).

Usage (from project root):
    python scripts/recompute_scores.py
"""

import asyncio
import os
import sys

# Project root on sys.path so `app.*` imports resolve when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# venv lives inside the project cwd; NLTK's import guard blocks `regex` otherwise.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

from dotenv import load_dotenv

load_dotenv()

from app.db.session import AsyncSessionLocal
from app.index.tfidf import recompute_tfidf_scores


async def main() -> None:
    async with AsyncSessionLocal() as session:
        await recompute_tfidf_scores(session)
    print("TF-IDF scores recomputed.")


if __name__ == "__main__":
    asyncio.run(main())
