import os

import pytest

from ragqa.embeddings import HashEmbedder
from ragqa.stores.pgvector import PgVectorStore
from ragqa.types import ChunkRecord


pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL is not set",
)


@pytest.fixture
def store() -> PgVectorStore:
    database_url = os.environ["DATABASE_URL"]
    created = PgVectorStore(database_url, dim=8, table="chunks_test")
    created.clear()
    yield created
    created.clear()
    created.close()


def test_pgvector_round_trip(store: PgVectorStore) -> None:
    embedder = HashEmbedder(dim=8)
    first = ChunkRecord(
        id="lighthouse:0",
        document_id="lighthouse",
        title="Lighthouse",
        source="lighthouse.md",
        text="The crown ring was cast in naval bronze.",
        embedding=embedder.embed_query("naval bronze crown ring"),
    )
    second = ChunkRecord(
        id="salt:0",
        document_id="salt",
        title="Salt",
        source="salt.md",
        text="The Dunn shipment was rejected for moisture.",
        embedding=embedder.embed_query("salt moisture Dunn shipment"),
    )
    store.upsert([first, second])
    assert store.count() == 2
    hits = store.query(first.embedding, k=1)
    assert hits[0].id == first.id
    assert hits[0].text == first.text
    assert hits[0].score > 0.9
    store.delete_document("lighthouse")
    assert [chunk.document_id for chunk in store.list_chunks()] == ["salt"]


def test_pgvector_rejects_a_different_dimension(store: PgVectorStore) -> None:
    del store
    database_url = os.environ["DATABASE_URL"]
    with pytest.raises(RuntimeError, match="dimension"):
        PgVectorStore(database_url, dim=16, table="chunks_test")
