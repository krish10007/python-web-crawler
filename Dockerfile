FROM python:3.11-slim

WORKDIR /app

# Build deps for packages that compile native extensions (e.g. asyncpg).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Slim API requirements (full requirements.txt also has torch/jupyter).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# tokenize() needs these corpora at import/runtime.
ENV NLTK_DISABLE_IMPORT_SECURITY=1
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"

COPY app/ ./app/

EXPOSE 8000
# Multiple worker processes so CPU-bound tokenize() (NLTK) in one request
# cannot stall the entire event loop for all concurrent clients.
# Rule of thumb: ~2 x CPU cores + 1 for mixed I/O+CPU APIs; 4 is a solid
# default for a small Docker Desktop / single-vCPU container without
# oversubscribing a tiny host.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
