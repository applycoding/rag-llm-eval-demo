"""Build the configured vector store."""

from __future__ import annotations

from typing import assert_never

from ragqa.config import Settings, resolved_database_url
from ragqa.embeddings import Embedder
from ragqa.stores.base import VectorStore
from ragqa.stores.memory import MemoryVectorStore
from ragqa.stores.pgvector import PgVectorStore
from ragqa.stores.pinecone import pinecone_store_from_settings
from ragqa.stores.weaviate import weaviate_store_from_settings
from ragqa.types import StoreName


def build_store(settings: Settings, embedder: Embedder) -> VectorStore:
    kind: StoreName = settings.vector_store
    if kind == "memory":
        return MemoryVectorStore()
    if kind == "pgvector":
        database_url = resolved_database_url(settings)
        if not database_url:
            raise RuntimeError("pgvector requires DATABASE_URL or RAGQA_DATABASE_URL.")
        return PgVectorStore(database_url, dim=embedder.dim)
    if kind == "pinecone":
        return pinecone_store_from_settings(settings)
    if kind == "weaviate":
        return weaviate_store_from_settings(settings)
    assert_never(kind)
