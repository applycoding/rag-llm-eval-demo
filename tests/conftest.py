from __future__ import annotations

from pathlib import Path

import pytest

from ragqa.config import Settings
from ragqa.service import QAService, build_service

ROOT = Path(__file__).resolve().parents[1]


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "embedder": "hash",
        "llm": "extractive",
        "vector_store": "memory",
        "auto_ingest": False,
        "corpus_dir": ROOT / "data" / "corpus",
        "prompt_path": ROOT / "prompts" / "qa.txt",
        "golden_path": ROOT / "data" / "golden" / "qa.json",
        "embedding_dim": 384,
        "chunk_size": 500,
        "chunk_overlap": 80,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def service(settings: Settings) -> QAService:
    built = build_service(settings)
    built.ingest_directory(settings.corpus_dir)
    return built
