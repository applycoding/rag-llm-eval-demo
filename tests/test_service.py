import pytest

from ragqa.llm import ABSTENTION
from ragqa.prompts import load_prompt
from ragqa.service import QAService, build_service
from ragqa.types import Document


def test_prompt_file_is_what_the_reader_sees(settings) -> None:
    captured: list[str] = []

    class Spy:
        name = "spy"

        def generate(self, prompt: str) -> str:
            captured.append(prompt)
            return "noted"

    built = build_service(settings)
    spy = Spy()
    built._langchain._llm.backend = spy
    built._llamaindex._llm.backend = spy
    built._tfidf_pipeline._llm = spy
    built.ingest_documents(
        [Document("note", "Note", "note.md", "The archive keeps a spare hawser in locker B.")]
    )
    for pipeline in ("langchain", "llamaindex", "tfidf"):
        built.query("Where is the spare hawser?", pipeline=pipeline, k=2)
    template = load_prompt(settings.prompt_path)
    assert "archivist of the Meridian Harbor Authority" in template
    assert ABSTENTION in template
    assert captured
    assert all("archivist of the Meridian Harbor Authority" in prompt for prompt in captured)
    assert all("Where is the spare hawser?" in prompt for prompt in captured)


def test_reingest_drops_orphan_chunks(service: QAService) -> None:
    before = service.store.count()
    service.ingest_documents(
        [Document("meridian-lighthouse", "Meridian Lighthouse", "meridian-lighthouse.md", "Short note.")]
    )
    remaining = [
        chunk for chunk in service.store.list_chunks() if chunk.document_id == "meridian-lighthouse"
    ]
    assert len(remaining) == 1
    assert service.store.count() < before


def test_each_pipeline_answers_from_the_archive(service: QAService) -> None:
    question = "What serial number does the compressed-air diaphone installed in 1911 carry?"
    for pipeline in ("langchain", "llamaindex", "tfidf"):
        answer = service.query(question, pipeline=pipeline, k=4)
        assert "DH-44" in answer.text
        assert any(chunk.document_id == "meridian-lighthouse" for chunk in answer.chunks)


def test_unknown_question_abstains(service: QAService) -> None:
    answer = service.query("What is the price of tea in the capital?", pipeline="tfidf", k=4)
    assert answer.text == ABSTENTION


def test_empty_question_is_rejected(service: QAService) -> None:
    with pytest.raises(ValueError):
        service.query("   ")
