from ragqa.eval.metrics import (
    answer_correctness,
    answer_relevance,
    context_recall,
    faithfulness,
    retrieval_hit,
)
from ragqa.eval.runner import load_golden, render_markdown_table, run_evaluation
from ragqa.llm import ABSTENTION
from ragqa.types import RetrievedChunk


def _chunk(document_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}:0",
        document_id=document_id,
        title=document_id,
        source=f"{document_id}.md",
        text=text,
        score=1.0,
    )


def test_faithfulness_rewards_supported_text_and_abstention() -> None:
    context = "The crown ring was cast in naval bronze."
    assert faithfulness(context, context) == 1.0
    assert faithfulness("The crown ring was launched into orbit in 1999.", context) < 0.5
    assert faithfulness(ABSTENTION, context) == 1.0


def test_relevance_and_correctness() -> None:
    question = "What was the crown ring cast in?"
    answer = "The crown ring was cast in naval bronze."
    assert answer_relevance(question, answer) > 0.5
    assert answer_relevance(question, ABSTENTION) == 0.0
    assert answer_correctness(answer, answer) == 1.0
    assert answer_correctness(answer, "unrelated text about pilots") < 0.3


def test_retrieval_metrics() -> None:
    chunks = [_chunk("lighthouse", "serial number DH-44")]
    assert retrieval_hit("lighthouse", chunks) is True
    assert retrieval_hit("salt", chunks) is False
    assert context_recall(["DH-44"], chunks) == 1.0
    assert context_recall(["missing"], chunks) == 0.0


def test_golden_set_matches_the_corpus(service, settings) -> None:
    items = load_golden(settings.golden_path)
    corpus = {chunk.document_id: [] for chunk in service.store.list_chunks()}
    for chunk in service.store.list_chunks():
        corpus.setdefault(chunk.document_id, []).append(chunk.text)
    for item in items:
        blob = "\n".join(corpus[item.document_id])
        assert item.reference_answer in blob
        for snippet in item.expected_snippets:
            assert snippet in blob


def test_golden_evaluation_clears_the_hit_rate_gate(service, settings) -> None:
    report = run_evaluation(service, load_golden(settings.golden_path), k=4)
    table = render_markdown_table(report)
    assert "| langchain |" in table
    assert "| llamaindex |" in table
    assert "| tfidf |" in table
    for summary in report.summaries:
        assert summary.retrieval_hit_rate == 1.0
        assert summary.context_recall == 1.0
        assert summary.faithfulness >= 0.9
        assert summary.answer_correctness >= 0.8
