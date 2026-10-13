"""Recursive character chunking with a preference for paragraph and sentence breaks."""

from __future__ import annotations

from ragqa.types import Chunk, Document

_SEPARATORS = ("\n\n", "\n", ". ", " ")


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = start + _break_offset(normalized[start:end])
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        else:
            next_start = _snap_to_boundary(normalized, next_start, end)
        start = next_start
    return chunks


def _snap_to_boundary(text: str, start: int, end: int) -> int:
    """Move a resume point forward to the next space when it lands inside a word."""
    if start <= 0 or start >= end:
        return start
    if text[start].isspace() or text[start - 1].isspace():
        return start
    for index in range(start, end):
        if text[index].isspace():
            return index + 1
    return start


def chunk_document(document: Document, *, chunk_size: int, overlap: int) -> list[Chunk]:
    pieces = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            id=f"{document.document_id}:{index}",
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            text=piece,
            index=index,
        )
        for index, piece in enumerate(pieces)
    ]


def _break_offset(window: str) -> int:
    midpoint = len(window) // 2
    for separator in _SEPARATORS:
        index = window.rfind(separator)
        if index > midpoint:
            return index + len(separator)
    return len(window)
