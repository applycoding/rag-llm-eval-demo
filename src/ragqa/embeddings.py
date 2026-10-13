"""Local and hosted embedding providers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol, assert_never

import numpy as np
from openai import OpenAI

from ragqa.config import Settings, openai_api_key

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Signed feature hashing.

    This is the offline embedder used by tests and CI. It needs no model download.
    Sentence-transformer embeddings are the local semantic provider.
    """

    name = "hash"

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in text.lower().split():
            cleaned = "".join(character for character in token if character.isalnum())
            if not cleaned:
                continue
            digest = hashlib.sha256(cleaned.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return [0.0] * self.dim
        return [float(value) for value in (vector / norm)]


class SentenceTransformerEmbedder:
    """Local semantic embeddings via sentence-transformers."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model: Any | None = None) -> None:
        self.model_name = model_name
        if model is None and SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                'Install it with pip install -e ".[embeddings]", or set RAGQA_EMBEDDER=hash.'
            )
        self._model = model if model is not None else SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(value) for value in row] for row in encoded]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """Hosted embeddings. Used only when an API key is configured."""

    name = "openai"

    def __init__(self, client: Any, model: str, dim: int) -> None:
        self._client = client
        self._model = model
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self.dim,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder
    if kind == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if kind == "openai":
        key = openai_api_key(settings)
        if not key:
            raise RuntimeError("OpenAI embeddings require OPENAI_API_KEY or RAGQA_OPENAI_API_KEY.")
        return OpenAIEmbedder(
            OpenAI(api_key=key),
            model=settings.openai_embedding_model,
            dim=settings.embedding_dim,
        )
    assert_never(kind)
