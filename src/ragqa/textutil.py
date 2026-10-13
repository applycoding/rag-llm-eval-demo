"""Token and similarity helpers shared by the reader and the evaluator."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "did",
        "does",
        "do",
    }
)


def content_terms(text: str) -> set[str]:
    terms = set()
    for token in _TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 2 or any(character.isdigit() for character in token):
            terms.add(token)
    return terms


def sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENTENCE.split(stripped) if part.strip()]


def term_recall(reference_terms: set[str], candidate_terms: set[str]) -> float:
    if not reference_terms:
        return 1.0
    return len(reference_terms & candidate_terms) / len(reference_terms)


def token_f1(reference: str, candidate: str) -> float:
    reference_terms = content_terms(reference)
    candidate_terms = content_terms(candidate)
    if not reference_terms and not candidate_terms:
        return 1.0
    if not reference_terms or not candidate_terms:
        return 0.0
    overlap = len(reference_terms & candidate_terms)
    precision = overlap / len(candidate_terms)
    recall = overlap / len(reference_terms)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)
