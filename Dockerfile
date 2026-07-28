FROM python:3.11-slim

WORKDIR /app

# System deps required by sentence-transformers/torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Embedding model cache lives here; mount as a volume in docker-compose so
# it survives container rebuilds and is only downloaded once.
ENV EMBEDDING_CACHE_DIR=/app/model_cache
RUN mkdir -p /app/model_cache /app/data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
