"""Shared records for ingestion, retrieval, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PipelineName = Literal["langchain", "llamaindex", "tfidf"]
StoreName = Literal["memory", "pgvector", "pinecone", "weaviate"]
EmbedderName = Literal["hash", "sentence-transformers", "openai"]
LLMName = Literal["auto", "extractive", "openai", "anthropic"]


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    source: str
    text: str
    index: int


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    title: str
    source: str
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    document_id: str
    title: str
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    pipeline: PipelineName
    chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    reference_answer: str
    document_id: str
    expected_snippets: list[str]
