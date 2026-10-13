from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragqa.embeddings import HashEmbedder
from ragqa.stores.memory import MemoryVectorStore
from ragqa.stores.pinecone import PineconeVectorStore, pinecone_store_from_settings
from ragqa.stores.weaviate import WeaviateVectorStore, stable_uuid, weaviate_store_from_settings
from ragqa.types import ChunkRecord
from tests.conftest import make_settings


def _record(text: str, document_id: str = "doc") -> ChunkRecord:
    embedding = HashEmbedder(dim=16).embed_query(text)
    return ChunkRecord(
        id=f"{document_id}:0",
        document_id=document_id,
        title=document_id,
        source=f"{document_id}.md",
        text=text,
        embedding=embedding,
    )


def test_memory_store_returns_the_nearest_chunk() -> None:
    store = MemoryVectorStore()
    bronze = _record("crown ring cast in naval bronze", "lighthouse")
    salt = _record("salt warehouse rejected the Dunn shipment", "salt")
    salt = ChunkRecord(
        id="salt:0",
        document_id="salt",
        title="salt",
        source="salt.md",
        text=salt.text,
        embedding=salt.embedding,
    )
    store.upsert([bronze, salt])
    hits = store.query(bronze.embedding, k=1)
    assert hits[0].document_id == "lighthouse"
    store.delete_document("lighthouse")
    assert store.count() == 1
    assert store.list_chunks()[0].document_id == "salt"


class FakePineconeIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, dict[str, object]] = {}
        self.deleted: list[dict[str, object]] = []

    def upsert(self, vectors: list[dict[str, object]], namespace: str) -> None:
        assert namespace == "ragqa"
        for vector in vectors:
            self.vectors[str(vector["id"])] = vector

    def query(self, vector: list[float], top_k: int, include_metadata: bool, namespace: str) -> object:
        del vector, namespace
        assert include_metadata is True
        matches = [
            SimpleNamespace(id=item["id"], score=0.8, metadata=item["metadata"])
            for item in self.vectors.values()
        ]
        return SimpleNamespace(matches=matches[:top_k])

    def delete(self, filter: dict[str, object] | None = None, namespace: str = "", delete_all: bool = False) -> None:
        self.deleted.append({"filter": filter, "namespace": namespace, "delete_all": delete_all})
        if delete_all:
            self.vectors.clear()

    def list(self, namespace: str) -> object:
        del namespace
        page = SimpleNamespace(vectors=[SimpleNamespace(id=key) for key in self.vectors])
        yield page

    def fetch(self, ids: list[str], namespace: str) -> object:
        del namespace
        vectors = {
            chunk_id: SimpleNamespace(id=chunk_id, metadata=self.vectors[chunk_id]["metadata"])
            for chunk_id in ids
        }
        return SimpleNamespace(vectors=vectors)


def test_pinecone_adapter_maps_records() -> None:
    index = FakePineconeIndex()
    store = PineconeVectorStore(index)
    record = _record("diaphone serial DH-44", "lighthouse")
    store.upsert([record])
    hits = store.query(record.embedding, k=2)
    assert hits[0].text == record.text
    assert hits[0].document_id == "lighthouse"
    assert store.list_chunks()[0].id == record.id
    store.delete_document("lighthouse")
    assert index.deleted[0]["filter"] == {"document_id": {"$eq": "lighthouse"}}
    store.clear()
    assert index.deleted[-1]["delete_all"] is True
    assert store.count() == 0


def test_pinecone_factory_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        pinecone_store_from_settings(make_settings())


class FakeWeaviateCollection:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.data = self
        self.query = self

    def insert(self, properties: dict[str, str], vector: list[float], uuid: str) -> None:
        self.objects[uuid] = {
            "properties": properties,
            "vector": vector,
            "uuid": uuid,
            "metadata": SimpleNamespace(distance=0.25),
        }

    def delete_by_id(self, uuid: str) -> None:
        self.objects.pop(str(uuid), None)

    def near_vector(self, **kwargs: object) -> object:
        limit = int(kwargs["limit"])
        objects = [
            SimpleNamespace(properties=item["properties"], uuid=item["uuid"], metadata=item["metadata"])
            for item in self.objects.values()
        ]
        return SimpleNamespace(objects=objects[:limit])

    def fetch_objects(self, limit: int, return_properties: list[str]) -> object:
        del limit, return_properties
        objects = [
            SimpleNamespace(properties=item["properties"], uuid=item["uuid"], metadata=None)
            for item in self.objects.values()
        ]
        return SimpleNamespace(objects=objects)


def test_weaviate_adapter_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragqa.stores.weaviate.MetadataQuery", None)
    collection = FakeWeaviateCollection()
    store = WeaviateVectorStore(collection)
    record = _record("Judge Inez Calder is the arbitrator", "ferry")
    record = ChunkRecord(
        id="ferry:0",
        document_id="ferry",
        title="ferry",
        source="ferry.md",
        text=record.text,
        embedding=record.embedding,
    )
    store.upsert([record])
    assert stable_uuid("ferry:0") in collection.objects
    hits = store.query(record.embedding, k=1)
    assert hits[0].text == record.text
    assert hits[0].score == pytest.approx(0.75)
    store.delete_document("ferry")
    assert store.count() == 0


def test_weaviate_factory_requires_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    with pytest.raises(RuntimeError, match="WEAVIATE_URL"):
        weaviate_store_from_settings(make_settings(weaviate_url=None))


def test_hosted_factories_require_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("WEAVIATE_URL", "http://localhost:8080")
    monkeypatch.setattr("ragqa.stores.pinecone.Pinecone", None)
    monkeypatch.setattr("ragqa.stores.weaviate.weaviate", None)
    with pytest.raises(RuntimeError, match="hosted"):
        pinecone_store_from_settings(make_settings(pinecone_index="archive"))
    with pytest.raises(RuntimeError, match="hosted"):
        weaviate_store_from_settings(make_settings())
