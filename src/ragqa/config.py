"""Locate project files and read process settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragqa.types import EmbedderName, LLMName, StoreName


def project_file(relative: str) -> Path:
    """Resolve a repo-relative file from the working directory or the source tree."""
    cwd_candidate = Path.cwd() / relative
    if cwd_candidate.exists():
        return cwd_candidate
    packaged = Path(__file__).resolve().parents[2] / relative
    if packaged.exists():
        return packaged
    return cwd_candidate


class Settings(BaseSettings):
    """Runtime configuration.

    Environment variables use the ``RAGQA_`` prefix. API keys are also read from
    the conventional ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` names, and the
    database URL is also read from ``DATABASE_URL``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vector_store: StoreName = "memory"
    database_url: str | None = None
    embedder: EmbedderName = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    llm: LLMName = "auto"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"
    openai_embedding_model: str = "text-embedding-3-small"
    corpus_dir: Path = Field(default_factory=lambda: project_file("data/corpus"))
    prompt_path: Path = Field(default_factory=lambda: project_file("prompts/qa.txt"))
    golden_path: Path = Field(default_factory=lambda: project_file("data/golden/qa.json"))
    auto_ingest: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 80
    pinecone_index: str | None = None
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None
    weaviate_collection: str = "RagqaChunk"
    weaviate_grpc_port: int = 50051


def openai_api_key(settings: Settings) -> str | None:
    return settings.openai_api_key or os.environ.get("OPENAI_API_KEY")


def anthropic_api_key(settings: Settings) -> str | None:
    return settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")


def resolved_database_url(settings: Settings) -> str | None:
    return settings.database_url or os.environ.get("DATABASE_URL")
