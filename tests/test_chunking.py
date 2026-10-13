import pytest

from ragqa.chunking import chunk_document, chunk_text
from ragqa.types import Document


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("Harbor log.", chunk_size=100, overlap=10) == ["Harbor log."]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("  \n", chunk_size=100, overlap=10) == []


def test_long_text_keeps_a_character_overlap() -> None:
    text = "abcdefghij" * 5
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert chunks[0] == text[:20]
    assert chunks[1] == text[15:35]
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)


def test_resume_does_not_start_mid_word() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunks = chunk_text(text, chunk_size=24, overlap=8)
    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert not chunk.startswith(" ")
        offset = text.find(chunk)
        assert offset > 0
        assert text[offset - 1].isspace()


def test_document_chunk_ids_are_stable() -> None:
    document = Document("log", "Log", "log.md", "One sentence. " * 40)
    chunks = chunk_document(document, chunk_size=40, overlap=8)
    assert chunks[0].id == "log:0"
    assert chunks[0].document_id == "log"
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
