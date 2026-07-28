"""
Ingestion pipeline: data/*.txt -> chunks -> embeddings -> Qdrant.

Flow (per the spec):
    TXT files -> Document Loader -> RecursiveCharacterTextSplitter
    -> Sentence Transformer embeddings -> Qdrant

Duplicate chunks (identical text, e.g. re-ingesting the same file) are
skipped by deriving a deterministic point ID from a hash of the chunk
text, and checking whether that ID already exists in the collection
before inserting.

Runnable standalone (`python ingest.py`) or imported and called from the
FastAPI /ingest endpoint.
"""

import hashlib
import uuid

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models as qmodels

from config import get_settings
from embeddings import get_embeddings
from models import IngestResponse
from qdrant_db import ensure_collection_exists, get_qdrant_client
from logging_config import get_logger

logger = get_logger()

# Stable namespace so the same chunk text always maps to the same UUID,
# which is how we detect + skip duplicates on re-ingestion.
_DEDUP_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _chunk_id(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_DEDUP_NAMESPACE, digest))


def load_documents(data_dir: str):
    """Load every .txt file from the data directory."""
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()


def run_ingestion() -> IngestResponse:
    """Run the full ingestion pipeline and return a summary."""
    settings = get_settings()
    ensure_collection_exists()

    documents = load_documents(str(settings.data_path))
    if not documents:
        logger.warning("No .txt files found in %s", settings.data_path)
        return IngestResponse(files_processed=0, chunks_created=0, chunks_inserted=0, chunks_skipped_duplicate=0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d source file(s) into %d chunk(s)", len(documents), len(chunks))

    embeddings = get_embeddings()
    client = get_qdrant_client()

    inserted, skipped = 0, 0
    points: list[qmodels.PointStruct] = []
    ids_in_batch: set[str] = set()

    for chunk in chunks:
        point_id = _chunk_id(chunk.page_content)

        if point_id in ids_in_batch:
            skipped += 1
            continue

        existing = client.retrieve(collection_name=settings.collection_name, ids=[point_id])
        if existing:
            skipped += 1
            continue

        vector = embeddings.embed_query(chunk.page_content)
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "page_content": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                },
            )
        )
        ids_in_batch.add(point_id)

    if points:
        client.upsert(collection_name=settings.collection_name, points=points)
        inserted = len(points)

    logger.info("Ingestion complete: inserted=%d skipped=%d", inserted, skipped)

    return IngestResponse(
        files_processed=len(documents),
        chunks_created=len(chunks),
        chunks_inserted=inserted,
        chunks_skipped_duplicate=skipped,
    )


if __name__ == "__main__":
    result = run_ingestion()
    logger.info("Ingestion summary: %s", result.model_dump_json())
