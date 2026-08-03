# Concurrent Web Crawler & TF-IDF Search Engine

An async Python web crawler and search stack built as a backend systems / algorithms portfolio project. It crawls the web concurrently with `asyncio` and `aiohttp`, deduplicates URLs with a Bloom filter, respects `robots.txt` and per-domain rate limits via Redis, builds a Postgres inverted index with TF-IDF scoring, and exposes search over a FastAPI API — all orchestrated with Docker Compose.

---

## Architecture

Crawl and search share one pipeline:

**Seed URLs** are enqueued on an `asyncio.Queue`. A **worker pool** pulls `(url, depth)` jobs, checks the **politeness layer** (`RobotsChecker` + Redis-backed `RateLimiter`), and consults an in-process **Bloom filter** (`UrlDeduplicator`) so the same URL is not scheduled twice. Allowed URLs are **fetched** with `aiohttp`, parsed with BeautifulSoup, and passed through **NLTK tokenization / stemming**. Each page is written into Postgres as `pages` / `terms` / `postings` rows (raw term frequencies first). After a crawl finishes, a **corpus-wide TF-IDF recompute** fills `document_frequency` and `tfidf_score`. Clients query the index through **`GET /search`** on FastAPI.

![Architecture](docs/diagrams/architecture.svg)

*Figure: end-to-end data flow from seed URLs through the worker pool, politeness/dedup layers, indexing, and the search API. (Diagram to be added at `docs/diagrams/architecture.svg`.)*

[`docs/screenshots/`](docs/screenshots/) contains real search-result screenshots from the running API.

---

## Tech stack

| Layer | Technologies |
|-------|----------------|
| Runtime | Python 3.11 |
| Crawler | `asyncio`, `aiohttp`, BeautifulSoup4 |
| Dedup | `pybloom-live` (`ScalableBloomFilter`) |
| NLP | NLTK (`word_tokenize`, Porter stemmer, stopwords) |
| Storage | PostgreSQL 16, Redis 7 |
| API / ORM | FastAPI, SQLAlchemy (async) + Alembic |
| Ops | Docker Compose, uvicorn (multi-worker) |
| Quality | pytest, pytest-asyncio, Locust |

---

## Key design decisions

### Why asyncio over threading

Crawling is dominated by waiting on the network. An `asyncio` worker pool plus a single shared `aiohttp.ClientSession` overlaps hundreds of in-flight HTTP requests without paying for one OS thread per connection. Shared crawl state (`CrawlStats`, the queue, the Bloom filter, Redis rate-limit keys) stays in one process with `asyncio.Lock` where atomicity matters, instead of coordinating locks across threads.

### Why a Bloom filter over a Python `set`

Exact URL membership in a `set` stores every string (plus CPython object and hash-table overhead) — typically on the order of **~100–200+ bytes per URL**. A Bloom filter at this project's default **0.1%** false-positive rate needs only ~10–15 bits per URL. At large crawl scale that is roughly **two to three orders of magnitude** less memory (commonly cited around **~1000×** vs a naïve set of full URL string objects). The trade-off is intentional: false positives may skip a rare new URL; false negatives never re-crawl a URL already seen.

### Why TF-IDF is computed in two passes

1. **`index_page`** (during the crawl) writes the `Page`, get-or-creates `Term` rows, and stores raw `term_frequency` with `tfidf_score = 0.0`.
2. **`recompute_tfidf_scores`** (after the crawl) sets each term's `document_frequency` and computes  
   `tf = term_frequency / word_count`,  
   `idf = max(0, ln(N / (1 + df)))`,  
   `tfidf_score = tf × idf`.

IDF is a **corpus-level** statistic. Until every page is indexed, \(N\) and each term's document frequency are unknown, so scores cannot be finalized on the hot crawl path.

### Why IDF is clamped at zero

On a small corpus, a common term can appear in nearly every document. Then `ln(N / (1 + df))` goes **negative**, so common words *subtract* from a page's score instead of contributing weakly. After observing negative scores in search results (e.g. queries like `domain name` on a ~20-page index), IDF was clamped with `max(0.0, …)` so ubiquitous terms contribute zero rather than punishing relevance.

### Diagnosing the p99 latency regression

Locust at **50 concurrent users** showed **p50 ≈ 13 ms** but **p99 ≈ 560–950 ms** (max ~1–2 s). The first hypothesis was SQLAlchemy pool exhaustion (defaults `pool_size=5`, `max_overflow=10` → max 15 connections). Expanding the pool to `20 + 30` **did not** fix the tail (p99 got worse while RPS barely moved).

Instrumenting `/search` separately timed `tokenize()` vs the DB query: under load, **`tokenize_ms` stayed ~0.1 ms** and **`query_ms` ~1–2 ms**, while Locust still reported multi-hundred-ms client latency. The bottleneck was a **single uvicorn worker**: sync NLTK work on the event loop serialized concurrent requests, so clients queued even when each handler's own timings looked fine. Switching the Docker CMD to **`--workers 4`** cut p99 to **~130 ms** and raised RPS to **~152** — the resume-ready number.

---

## Benchmark results

### Crawl throughput

Multi-domain seeds (Wikipedia returns HTTP 403 for this bot traffic), `max_depth=3`, ~1 req/s per domain via Redis rate limiting. Worker-scaling runs used `max_pages=300`; a longer single run used `max_pages=1000`.

| Workers | Pages Crawled | Elapsed (s) | Pages/min |
|---------|---------------|-------------|-----------|
| 10 | 300 | 664.33 | 27.10 |
| 25 | 300 | 484.55 | 37.15 |
| 50 | 300 | 378.04 | 47.61 |
| 50 | 1000 | 4855.07 | 12.36 |

Throughput dropped from **47.61 pages/min** (300-page run) to **12.36 pages/min** (1000-page run) for a real scheduling reason, not a worker-pool failure: late in the longer crawl, BFS discovery concentrated the queue on a single high-link-density domain (`rfc-editor.org`), so per-domain rate limiting (~1 req/s) became the dominant bottleneck rather than concurrency. At the scale discussed in [Scaling toward 10M pages](#scaling-toward-10m-pages), queue scheduling would need to prevent one domain from starving others of worker attention (e.g. round-robin across domains or per-domain queue caps).

Final inverted index after the 1000-page crawl (and TF-IDF recompute): **986 pages**, **23,364 terms**, **268,148 postings**.

### Search API latency

Locust: 50 concurrent users, spawn rate 10/s, 60 s, queries drawn from real high-`document_frequency` stems in Postgres.

| Config | p50 | p99 | RPS | Failure rate |
|--------|-----|-----|-----|--------------|
| Single uvicorn worker, default DB pool | 13 ms | 560 ms | 127.3 | 0% |
| Single worker, expanded DB pool (20+30) | 11 ms | 950 ms | 138.9 | 0% |
| **4 uvicorn workers, expanded pool** | **8 ms** | **130 ms** | **152.3** | **0%** |

Locust’s p50/p99 figures above were measured under **sustained load against already-warm** uvicorn workers. A single cold request immediately after deployment measures higher (~50–70 ms) because of one-time SQLAlchemy query compilation and connection-pool initialization on that worker. Instrumented timing plus `EXPLAIN ANALYZE` confirmed the raw SQL still executes in **~3.7 ms** even at the full **268K-posting** corpus size — indexes are working correctly; the visible cold latency is Python/ORM startup overhead, which is standard for ORM-based services rather than a query-performance bug.

---

## Getting started

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local scripts / tests)
- Copy env template and adjust ports if needed (this machine often maps Postgres to **5434** when native Postgres occupies 5432)

```bash
cp .env.example .env
# Edit DATABASE_URL / POSTGRES_PORT if 5432 is already taken on the host
```

### Start infrastructure + API

```bash
docker compose up -d --build
```

This starts Postgres, Redis, and the FastAPI service on **http://localhost:8000**.

Apply migrations (from the project root, with the venv and `.env` loaded):

```bash
python -m alembic upgrade head
```

One-time NLTK data (for local crawls / tests):

```bash
python scripts/download_nltk_data.py
```

### Run a crawl

Throughput benchmark (fresh `UrlDeduplicator` / queue per worker-count run):

```bash
# On Windows, if NLTK complains about cwd imports:
# set NLTK_DISABLE_IMPORT_SECURITY=1
python scripts/benchmark_crawl.py
```

Or call the library API from your own script:

```python
from app.crawler.core import crawl

stats = await crawl(
    seed_urls=["https://example.com", "https://www.python.org/"],
    num_workers=10,
    max_pages=100,
    max_depth=2,
)
```

After indexing, recompute TF-IDF without re-crawling:

```bash
python scripts/recompute_scores.py
```

### Query the search API

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/search?q=python+documentation&limit=3"
```

---

## Example search queries

Terms below are **real stems** from the live `terms` table after the full multi-domain crawl (**986 pages** / **23,364 terms** / **268,148 postings**), e.g. `document` (df=488), `content` (459), `search` (453), `develop` (356), plus `python`-related pages in the index.

```bash
curl -s "http://localhost:8000/search?q=python+documentation&limit=5"
```

Example response (captured live against the full 1000-page corpus):

```json
{
  "results": [
    {
      "url": "https://scikit-learn.org/",
      "title": "scikit-learn: machine learning in Python — scikit-learn 0.16.1 documentation",
      "score": 0.542500050562249
    },
    {
      "url": "https://www.python.org/doc/versions/",
      "title": "Python documentation by version | Python.org",
      "score": 0.4155116929351436
    },
    {
      "url": "https://docs.python.org/3/",
      "title": "3.14.6 Documentation",
      "score": 0.34607948529704113
    },
    {
      "url": "https://docs.python.org/3.16/",
      "title": "3.16.0a0 Documentation",
      "score": 0.34607948529704113
    },
    {
      "url": "https://docs.python.org/3.15/",
      "title": "3.15.0b4 Documentation",
      "score": 0.34607948529704113
    }
  ],
  "query_time_ms": 4.299
}
```

Other realistic queries against this corpus: `search`, `document`, `content`, `python`, `license` (stemmed to match indexed tokens).

Real search-result screenshots live under [`docs/screenshots/`](docs/screenshots/).

---

## Testing

The suite under `app/tests/` currently includes **12 tests**:

- Bloom filter dedup (including a bounded false-positive-rate check)
- Tokenizer (stopwords, stemming, non-alphabetic filtering)
- TF-IDF (`compute_term_frequencies` + integration proving rare terms outrank common ones; non-negative scores)
- FastAPI `/health` and `/search` (dependency-overridden session)

Tests use an **isolated `crawler_db_test`** database derived from `DATABASE_URL` (created/dropped per pytest session) so they never touch the production crawl data in `crawler_db`.

```bash
pytest -v
```

Load testing:

```bash
locust -f scripts/locustfile.py --headless -u 50 -r 10 -t 60s \
  --host http://localhost:8000 --csv scripts/results/loadtest
```

---

## Project structure

```
app/
├── api/          # FastAPI app: /health, /search
├── crawler/      # Worker pool, fetcher, Bloom dedup, politeness (robots + Redis)
├── db/           # SQLAlchemy models (Page, Term, Posting) and async session
├── index/        # Text extraction, NLTK tokenize/stem, TF-IDF index + recompute
└── tests/        # pytest suite (isolated crawler_db_test)
alembic/          # Migrations
scripts/          # Benchmarks, Locust, NLTK download, TF-IDF recompute utility
docs/
├── diagrams/     # architecture.svg (Excalidraw export)
└── screenshots/  # Search UI / response screenshots
```

---

## Scaling toward 10M pages

This codebase is intentionally a **single-process** design for clarity. At much larger scale you would typically:

- Replace the in-process `asyncio.Queue` with a **distributed queue** (Redis Streams, SQS, Kafka) so many crawler hosts can share work
- Move URL dedup from an in-process Bloom filter to a **shared store** (Redis Bloom / Redis Set) so instances agree on “seen”
- **Shard** the Postgres inverted index (or move hot postings to a search engine such as OpenSearch) as term/page cardinality grows
- Keep politeness global: one Redis rate-limit keyspace and robots cache shared across the fleet
- Run API replicas behind a load balancer; keep uvicorn multi-worker (or an async pool) so CPU-bound tokenization does not serialize traffic on one event loop

---

## License

Educational / portfolio project.
