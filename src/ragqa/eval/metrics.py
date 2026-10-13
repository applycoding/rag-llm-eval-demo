"""Offline RAGAS-style metrics.

Faithfulness, answer relevance, and retrieval hit rate are computed without an
LLM judge so the suite runs when no API key is set. Answer correctness (token
F1 against the reference) and context recall are reported alongside them.
"""

from __future__ import annotations

from ragqa.llm import ABSTENTION
from ragqa.textutil import content_terms, sentences, term_recall, token_f1
from ragqa.types import RetrievedChunk

_SUPPORT_THRESHOLD = 0.6


def faithfulness(answer: str, context: str) -> float:
    """Fraction of answer sentences whose content words are supported by the context."""
    if answer.strip() == ABSTENTION:
        return 1.0
    claims = sentences(answer)
    if not claims:
        return 0.0
    context_terms = content_terms(context)
    supported = 0
    for claim in claims:
        claim_terms = content_terms(claim)
        if not claim_terms or term_recall(claim_terms, context_terms) >= _SUPPORT_THRESHOLD:
            supported += 1
    return supported / len(claims)


def answer_relevance(question: str, answer: str) -> float:
    """Recall of the question's content words in the answer.

    This is the offline stand-in for RAGAS answer relevancy, which normally
    asks an LLM to reverse-generate questions from the answer.
    """
    if answer.strip() == ABSTENTION:
        return 0.0
    return term_recall(content_terms(question), content_terms(answer))


def answer_correctness(reference: str, answer: str) -> float:
    return token_f1(reference, answer)


def retrieval_hit(document_id: str, chunks: list[RetrievedChunk]) -> bool:
    return any(chunk.document_id == document_id for chunk in chunks)


def context_recall(snippets: list[str], chunks: list[RetrievedChunk]) -> float:
    if not snippets:
        return 1.0
    blob = "\n".join(chunk.text for chunk in chunks).lower()
    found = sum(1 for snippet in snippets if snippet.lower() in blob)
    return found / len(snippets)
