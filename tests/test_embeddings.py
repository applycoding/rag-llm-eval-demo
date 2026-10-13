import pytest

from ragqa.embeddings import HashEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashEmbedder(dim=32)
    first = embedder.embed_query("naval bronze crown ring")
    second = embedder.embed_query("naval bronze crown ring")
    assert first == second
    assert len(first) == 32
    norm = sum(value * value for value in first) ** 0.5
    assert norm == pytest.approx(1.0)
    assert embedder.embed_documents(["alpha", "beta"])[0] != embedder.embed_documents(["beta"])[0]


def test_hash_embedder_rejects_bad_dim() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_sentence_transformer_uses_injected_model() -> None:
    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts: list[str], normalize_embeddings: bool = False) -> list[list[float]]:
            assert normalize_embeddings is True
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", model=FakeModel())
    assert embedder.dim == 4
    assert embedder.embed_query("lantern") == [0.1, 0.2, 0.3, 0.4]
    assert embedder.name == "sentence-transformers"


def test_sentence_transformer_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragqa.embeddings.SentenceTransformer", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")


def test_openai_embedder_preserves_index_order() -> None:
    class Item:
        def __init__(self, index: int, embedding: list[float]) -> None:
            self.index = index
            self.embedding = embedding

    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

            class Embeddings:
                def __init__(self, outer: Client) -> None:
                    self._outer = outer

                def create(self, **kwargs: object) -> object:
                    self._outer.calls.append(kwargs)
                    return type("Response", (), {"data": [Item(1, [3.0]), Item(0, [1.0])]})()

            self.embeddings = Embeddings(self)

    client = Client()
    embedder = OpenAIEmbedder(client, model="text-embedding-3-small", dim=8)
    assert embedder.embed_documents(["a", "b"]) == [[1.0], [3.0]]
    assert client.calls[0]["dimensions"] == 8
    assert client.calls[0]["model"] == "text-embedding-3-small"
