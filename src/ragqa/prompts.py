"""File-backed prompt templates."""

from __future__ import annotations

from pathlib import Path

from ragqa.types import RetrievedChunk


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if "{context}" not in text or "{question}" not in text:
        raise ValueError(f"{path} must contain {{context}} and {{question}} placeholders")
    return text


def render_prompt(template: str, *, context: str, question: str) -> str:
    return template.format(context=context, question=question)


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no matching passages)"
    blocks = [
        f"[{index}] ({chunk.document_id}).\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)
