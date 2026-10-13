"""Load corpus files into documents."""

from __future__ import annotations

import re
from pathlib import Path

from ragqa.types import Document

_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def load_directory(path: Path) -> list[Document]:
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {path}")
    documents = []
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() not in {".md", ".txt"} or not file_path.is_file():
            continue
        documents.append(load_file(file_path))
    return documents


def load_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    heading = _HEADING.search(text)
    title = heading.group(1).strip() if heading else path.stem
    return Document(document_id=path.stem, title=title, source=path.name, text=text)
