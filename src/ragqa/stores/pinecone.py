"""Pinecone adapter over the shared vector-store methods."""

from __future__ import annotations

import os
from typing import Any

from ragqa.config import Settings

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None  # type: ignore[misc, assignment]
from ragqa.types import ChunkRecord, RetrievedChunk


class PineconeVectorStore:
    """Map chunk records onto a Pinecone index.

    The index object is injected so tests can exercise the mapping without the
    hosted service. ``pinecone_store_from_settings`` builds a real client.
    """

    def __init__(self, index: Any, namespace: str = "ragqa") -> None:
        self._index = index
        self._namespace = namespace

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        vectors = [
            {
                "id": record.id,
                "values": list(record.embedding),
                "metadata": {
                    "document_id": record.document_id,
                    "title": record.title,
                    "source": record.source,
                    "text": record.text,
                },
            }
            for record in records
        ]
        self._index.upsert(vectors=vectors, namespace=self._namespace)

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        if k < 1:
            return []
        response = self._index.query(
            vector=list(embedding),
            top_k=k,
            include_metadata=True,
            namespace=self._namespace,
        )
        matches = _get(response, "matches") or []
        chunks = []
        for match in matches:
            metadata = _get(match, "metadata") or {}
            chunks.append(
                RetrievedChunk(
                    id=str(_get(match, "id")),
                    document_id=str(_get(metadata, "document_id") or ""),
                    title=str(_get(metadata, "title") or ""),
                    source=str(_get(metadata, "source") or ""),
                    text=str(_get(metadata, "text") or ""),
                    score=float(_get(match, "score") or 0.0),
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        self._index.delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=self._namespace,
        )

    def list_chunks(self) -> list[RetrievedChunk]:
        ids = _listed_ids(self._index.list(namespace=self._namespace))
        chunks: list[RetrievedChunk] = []
        for start in range(0, len(ids), 100):
            fetched = self._index.fetch(ids=ids[start : start + 100], namespace=self._namespace)
            vectors = _get(fetched, "vectors") or {}
            items = (
                vectors.items()
                if isinstance(vectors, dict)
                else [(_get(vector, "id"), vector) for vector in vectors]
            )
            for chunk_id, vector in items:
                metadata = _get(vector, "metadata") or {}
                chunks.append(
                    RetrievedChunk(
                        id=str(chunk_id),
                        document_id=str(_get(metadata, "document_id") or ""),
                        title=str(_get(metadata, "title") or ""),
                        source=str(_get(metadata, "source") or ""),
                        text=str(_get(metadata, "text") or ""),
                        score=0.0,
                    )
                )
        return sorted(chunks, key=lambda chunk: chunk.id)

    def count(self) -> int:
        return len(self.list_chunks())

    def clear(self) -> None:
        self._index.delete(delete_all=True, namespace=self._namespace)

    def close(self) -> None:
        return None


def pinecone_store_from_settings(settings: Settings) -> PineconeVectorStore:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = settings.pinecone_index or os.environ.get("PINECONE_INDEX")
    if not api_key or not index_name:
        raise RuntimeError(
            "Pinecone requires PINECONE_API_KEY and PINECONE_INDEX (or RAGQA_PINECONE_INDEX)."
        )
    if Pinecone is None:
        raise RuntimeError('Install the Pinecone adapter with pip install -e ".[hosted]".')
    index = Pinecone(api_key=api_key).Index(index_name)
    return PineconeVectorStore(index)


def _listed_ids(pages: Any) -> list[str]:
    ids: list[str] = []
    for page in pages:
        if isinstance(page, str):
            ids.append(page)
            continue
        vectors = _get(page, "vectors")
        if vectors:
            for item in vectors:
                identifier = item if isinstance(item, str) else _get(item, "id")
                if identifier:
                    ids.append(str(identifier))
            continue
        page_ids = _get(page, "ids")
        if page_ids:
            ids.extend(str(item) for item in page_ids)
    return ids


def _get(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
