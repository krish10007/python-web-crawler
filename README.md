# Concurrent Web Crawler & TF-IDF Search Engine

Async Python crawler and search stack for a backend systems / algorithms portfolio project. Workers crawl concurrently with `asyncio` and `aiohttp`, deduplicate URLs with a Bloom filter, honor `robots.txt` and Redis-backed per-domain rate limits, build a Postgres inverted index scored with TF-IDF, and serve results from a FastAPI app. Everything runs under Docker Compose.

---

## Architecture

Seed URLs go onto an `asyncio.Queue`. A worker pool pulls `(url, depth)` jobs, runs them through the politeness layer (`RobotsChecker` + Redis `RateLimiter`), and checks an in-process Bloom filter (`UrlDeduplicator`) so the same URL is not scheduled twice. Allowed pages are fetched with `aiohttp`, parsed with BeautifulSoup, tokenized/stemmed with NLTK, and written to Postgres as `pages` / `terms` / `postings` (raw term frequencies first). After the crawl, a corpus-wide TF-IDF recompute fills `document_frequency` and `tfidf_score`. Clients hit `GET /search` on FastAPI.

![Architecture](docs/diagrams/architecture.svg)

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

Crawling spends most of its time waiting on the network. An `asyncio` worker pool with one shared `aiohttp.ClientSession` keeps a lot of HTTP requests in flight without a thread per connection. Crawl state (`CrawlStats`, the queue, the Bloom filter, Redis rate-limit keys) stays in one process; `asyncio.Lock` covers the few places that need atomic updates. That is simpler than sharing the same structures across threads.

### Why a Bloom filter over a Python `set`

A `set` of URL strings costs roughly ~100–200+ bytes per URL once you count CPython object and hash-table overhead. The Bloom filter here targets a 0.1% false-positive rate and needs only ~10–15 bits per URL, which is about ~1000× less memory at large crawl scale (two to three orders of magnitude). The trade-off is deliberate: a false positive might skip a rare new URL; a false negative never re-crawls something already seen.

### Why TF-IDF is computed in two passes

1. `index_page` (during the crawl) writes the `Page`, get-or-creates `Term` rows, and stores raw `term_frequency` with `tfidf_score = 0.0`.
2. `recompute_tfidf_scores` (after the crawl) sets each term's `document_frequency` and computes  
   `tf = term_frequency / word_count`,  
   `idf = max(0, ln(N / (1 + df)))`,  
   `tfidf_score = tf × idf`.

IDF depends on the whole corpus. Until indexing finishes, \(N\) and each term's document frequency are unknown, so final scores cannot be written on the hot crawl path.

### Why IDF is clamped at zero

On a small corpus a common term can show up in almost every document, so `ln(N / (1 + df))` goes negative and those terms drag scores down instead of adding a little weight. I hit that on queries like `domain name` against a ~20-page index. Clamping with `max(0.0, …)` makes ubiquitous terms contribute zero instead of hurting relevance.

### Diagnosing the p99 latency regression

Locust at 50 concurrent users showed p50 ≈ 13 ms but p99 ≈ 560–950 ms (max ~1–2 s). I first blamed SQLAlchemy pool exhaustion (defaults `pool_size=5`, `max_overflow=10` → max 15 connections). Growing the pool to `20 + 30` did not fix the tail; p99 got worse and RPS barely moved.

I instrumented `/search` to time `tokenize()` and the DB query separately. Under load, `tokenize_ms` stayed ~0.1 ms and `query_ms` ~1–2 ms, while Locust still saw multi-hundred-ms client latency. The problem was a single uvicorn worker: sync NLTK on the event loop serialized concurrent requests, so clients queued even when each handler looked fast. Switching the Docker CMD to `--workers 4` brought p99 down to ~130 ms and RPS up to ~152.

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

Throughput fell from 47.61 pages/min on the 300-page run to 12.36 pages/min on the 1000-page run for a scheduling reason, not because the worker pool failed. Late in the longer crawl, BFS discovery piled the queue onto one high-link-density domain (`rfc-editor.org`), so the ~1 req/s per-domain cap became the bottleneck instead of concurrency. At the scale discussed in [Scaling toward 10M pages](#scaling-toward-10m-pages), the queue would need fairer scheduling across domains (round-robin or per-domain caps) so one host cannot starve the rest.

Final inverted index after the 1000-page crawl (and TF-IDF recompute): 986 pages, 23,364 terms, 268,148 postings.

### Search API latency

Locust: 50 concurrent users, spawn rate 10/s, 60 s, queries drawn from real high-`document_frequency` stems in Postgres.

| Config | p50 | p99 | RPS | Failure rate |
|--------|-----|-----|-----|--------------|
| Single uvicorn worker, default DB pool | 13 ms | 560 ms | 127.3 | 0% |
| Single worker, expanded DB pool (20+30) | 11 ms | 950 ms | 138.9 | 0% |
| **4 uvicorn workers, expanded pool** | **8 ms** | **130 ms** | **152.3** | **0%** |

Those Locust numbers are under sustained load on already-warm uvicorn workers. A single cold request right after deploy is higher (~50–70 ms) from one-time SQLAlchemy query compilation and connection-pool setup on that worker. Instrumented timings plus `EXPLAIN ANALYZE` show the raw SQL still runs in ~3.7 ms against the full 268K-posting corpus, so the indexes are doing their job. The cold bump is Python/ORM startup cost, not a slow query plan.

---

## Getting started

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local scripts / tests)
- Copy the env template and adjust ports if needed (this machine often maps Postgres to 5434 when native Postgres already owns 5432)

```bash
cp .env.example .env
# Edit DATABASE_URL / POSTGRES_PORT if 5432 is already taken on the host
```

### Start infrastructure + API

```bash
docker compose up -d --build
```

This starts Postgres, Redis, and the FastAPI service on http://localhost:8000.

Apply migrations from the project root (venv and `.env` loaded):

```bash
python -m alembic upgrade head
```

One-time NLTK data for local crawls / tests:

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

Stems below come from the live `terms` table after the full multi-domain crawl (986 pages / 23,364 terms / 268,148 postings), e.g. `document` (df=488), `content` (459), `search` (453), `develop` (356), plus `python`-related pages in the index.

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

Other queries that hit this corpus well: `search`, `document`, `content`, `python`, `license` (stemmed to match indexed tokens).

---

## Testing

`app/tests/` has 12 tests covering:

- Bloom filter dedup (including a bounded false-positive-rate check)
- Tokenizer (stopwords, stemming, non-alphabetic filtering)
- TF-IDF (`compute_term_frequencies` + integration proving rare terms outrank common ones; non-negative scores)
- FastAPI `/health` and `/search` (dependency-overridden session)

Tests use an isolated `crawler_db_test` database derived from `DATABASE_URL` (created/dropped per pytest session) so they never touch crawl data in `crawler_db`.

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

This stays a single-process design on purpose so the code is easy to follow. At much larger scale you would typically:

- Replace the in-process `asyncio.Queue` with a distributed queue (Redis Streams, SQS, Kafka) so many crawler hosts can share work
- Move URL dedup from an in-process Bloom filter to a shared store (Redis Bloom / Redis Set) so instances agree on “seen”
- Shard the Postgres inverted index (or move hot postings to something like OpenSearch) as term/page cardinality grows
- Keep politeness global: one Redis rate-limit keyspace and robots cache shared across the fleet
- Run API replicas behind a load balancer; keep uvicorn multi-worker (or an async pool) so CPU-bound tokenization does not serialize traffic on one event loop

---

## License

Educational / portfolio project.
