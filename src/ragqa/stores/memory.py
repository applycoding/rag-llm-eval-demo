"""In-process cosine store used by tests and by runs without PostgreSQL."""

from __future__ import annotations

from ragqa.textutil import cosine
from ragqa.types import ChunkRecord, RetrievedChunk


class MemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, ChunkRecord] = {}

    def upsert(self, records: list[ChunkRecord]) -> None:
        for record in records:
            self._records[record.id] = record

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        if k < 1 or not self._records:
            return []
        ranked = sorted(
            (
                (
                    cosine(embedding, record.embedding),
                    record,
                )
                for record in self._records.values()
            ),
            key=lambda item: (-item[0], item[1].id),
        )
        return [_with_score(record, score) for score, record in ranked[:k]]

    def delete_document(self, document_id: str) -> None:
        doomed = [key for key, record in self._records.items() if record.document_id == document_id]
        for key in doomed:
            del self._records[key]

    def list_chunks(self) -> list[RetrievedChunk]:
        records = sorted(self._records.values(), key=lambda record: record.id)
        return [_with_score(record, 0.0) for record in records]

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    def close(self) -> None:
        return None


def _with_score(record: ChunkRecord, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=record.id,
        document_id=record.document_id,
        title=record.title,
        source=record.source,
        text=record.text,
        score=score,
    )
