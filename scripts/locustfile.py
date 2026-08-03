"""
Locust load test for GET /search against a real inverted-index vocabulary.

Query terms were taken from the live `terms` table ordered by
document_frequency DESC (stemmed index tokens), so requests hit real
postings rather than empty result sets.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Top document_frequency stems from crawler_db.terms (as of load-test setup).
SEARCH_TERMS = [
    "document",
    "page",
    "use",
    "code",
    "content",
    "report",
    "inform",
    "help",
    "develop",
    "websit",
    "list",
    "new",
    "book",
    "search",
    "commun",
    "contact",
    "licens",
    "includ",
    "get",
    "copyright",
]


class SearchUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def search(self) -> None:
        query = random.choice(SEARCH_TERMS)
        with self.client.get(
            "/search",
            params={"q": query, "limit": 10},
            name="/search?q=[term]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return
            try:
                body = response.json()
            except Exception as exc:
                response.failure(f"invalid json: {exc}")
                return
            if "results" not in body or "query_time_ms" not in body:
                response.failure("missing results/query_time_ms")
                return
            response.success()
