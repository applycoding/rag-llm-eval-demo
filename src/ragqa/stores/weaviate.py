"""Weaviate adapter over the shared vector-store methods."""

from __future__ import annotations

import os
import uuid
from typing import Any
from urllib.parse import urlparse

from ragqa.config import Settings
from ragqa.types import ChunkRecord, RetrievedChunk

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
except ImportError:
    weaviate = None  # type: ignore[misc, assignment]
    Configure = None  # type: ignore[misc, assignment]
    DataType = None  # type: ignore[misc, assignment]
    Property = None  # type: ignore[misc, assignment]
    Auth = None  # type: ignore[misc, assignment]
    MetadataQuery = None  # type: ignore[misc, assignment]

_NAMESPACE = uuid.UUID("8f3d3c0e-6a1b-4f0a-9c2d-7b1e5a4c8d20")
_PROPERTIES = ["chunk_id", "document_id", "title", "source", "text"]


class WeaviateVectorStore:
    """Map chunk records onto a Weaviate collection that brings its own vectors.

    The collection object is injected. ``weaviate_store_from_settings`` builds a
    real v4 client and creates the collection when it is missing.
    """

    def __init__(self, collection: Any, *, client: Any | None = None) -> None:
        self._collection = collection
        self._client = client

    def upsert(self, records: list[ChunkRecord]) -> None:
        for record in records:
            self._collection.data.insert(
                properties=_properties(record),
                vector=list(record.embedding),
                uuid=stable_uuid(record.id),
            )

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        if k < 1:
            return []
        arguments: dict[str, Any] = {
            "near_vector": list(embedding),
            "limit": k,
            "return_properties": list(_PROPERTIES),
        }
        metadata = self._distance_metadata()
        if metadata is not None:
            arguments["return_metadata"] = metadata
        result = self._collection.query.near_vector(**arguments)
        objects = _get(result, "objects") or []
        return [_from_object(obj) for obj in objects]

    def delete_document(self, document_id: str) -> None:
        for obj in self._objects():
            properties = _get(obj, "properties") or {}
            if _get(properties, "document_id") == document_id:
                self._collection.data.delete_by_id(_get(obj, "uuid"))

    def list_chunks(self) -> list[RetrievedChunk]:
        return sorted((_from_object(obj) for obj in self._objects()), key=lambda chunk: chunk.id)

    def count(self) -> int:
        return len(self.list_chunks())

    def clear(self) -> None:
        for obj in self._objects():
            self._collection.data.delete_by_id(_get(obj, "uuid"))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _objects(self) -> list[Any]:
        result = self._collection.query.fetch_objects(
            limit=10_000,
            return_properties=list(_PROPERTIES),
        )
        return list(_get(result, "objects") or [])

    def _distance_metadata(self) -> Any:
        if MetadataQuery is None:
            return None
        return MetadataQuery(distance=True)


def weaviate_store_from_settings(settings: Settings) -> WeaviateVectorStore:
    url = settings.weaviate_url or os.environ.get("WEAVIATE_URL")
    if not url:
        raise RuntimeError("Weaviate requires WEAVIATE_URL or RAGQA_WEAVIATE_URL.")
    if weaviate is None or Configure is None or DataType is None or Property is None or Auth is None:
        raise RuntimeError('Install the Weaviate adapter with pip install -e ".[hosted]".')
    parsed = urlparse(url)
    if not parsed.hostname:
        raise RuntimeError(f"WEAVIATE_URL is not a valid URL: {url}")
    api_key = settings.weaviate_api_key or os.environ.get("WEAVIATE_API_KEY")
    auth = Auth.api_key(api_key) if api_key else None
    secure = parsed.scheme == "https"
    client = weaviate.connect_to_custom(
        http_host=parsed.hostname,
        http_port=parsed.port or (443 if secure else 8080),
        http_secure=secure,
        grpc_host=parsed.hostname,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=secure,
        auth_credentials=auth,
    )
    name = settings.weaviate_collection
    if not client.collections.exists(name):
        client.collections.create(
            name=name,
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(),
        )
    collection = client.collections.get(name)
    return WeaviateVectorStore(collection, client=client)


def stable_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _properties(record: ChunkRecord) -> dict[str, str]:
    return {
        "chunk_id": record.id,
        "document_id": record.document_id,
        "title": record.title,
        "source": record.source,
        "text": record.text,
    }


def _from_object(obj: Any) -> RetrievedChunk:
    properties = _get(obj, "properties") or {}
    metadata = _get(obj, "metadata")
    distance = _get(metadata, "distance") if metadata is not None else None
    score = 0.0 if distance is None else 1.0 - float(distance)
    return RetrievedChunk(
        id=str(_get(properties, "chunk_id") or _get(obj, "uuid") or ""),
        document_id=str(_get(properties, "document_id") or ""),
        title=str(_get(properties, "title") or ""),
        source=str(_get(properties, "source") or ""),
        text=str(_get(properties, "text") or ""),
        score=score,
    )


def _get(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
