"""TF-IDF retriever. Same prompt and reader as the dense pipelines, different ranker."""

from __future__ import annotations

from dataclasses import replace

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragqa.llm import LanguageModel
from ragqa.prompts import format_context, render_prompt
from ragqa.types import RetrievedChunk


class TfidfRetriever:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
        self._matrix = None
        self._chunks: list[RetrievedChunk] = []

    def fit(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            self._matrix = None
            self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
            return
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in chunks])

    def query(self, question: str, k: int) -> list[RetrievedChunk]:
        if k < 1 or self._matrix is None or not self._chunks:
            return []
        scores = cosine_similarity(self._vectorizer.transform([question]), self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            replace(self._chunks[int(index)], score=float(scores[int(index)]))
            for index in order
            if scores[int(index)] > 0.0
        ]


class TfidfPipeline:
    name = "tfidf"

    def __init__(self, retriever: TfidfRetriever, llm: LanguageModel, template: str) -> None:
        self._retriever = retriever
        self._llm = llm
        self._template = template

    def answer(self, question: str, k: int) -> tuple[str, list[RetrievedChunk]]:
        chunks = self._retriever.query(question, k)
        prompt = render_prompt(self._template, context=format_context(chunks), question=question)
        return self._llm.generate(prompt), chunks
