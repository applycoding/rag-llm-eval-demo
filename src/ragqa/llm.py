"""Readers: an offline extractive model, plus optional OpenAI and Anthropic chat models."""

from __future__ import annotations

import re
from typing import Any, Literal, Protocol, assert_never

from anthropic import Anthropic
from openai import OpenAI

from ragqa.config import Settings, anthropic_api_key, openai_api_key
from ragqa.textutil import content_terms, sentences, term_recall
from ragqa.types import LLMName

ABSTENTION = "The archive does not contain enough information to answer that."
_CITATION = re.compile(r"^(?:\[\d+\] \([^)]+\)\.\s*)+")


class LanguageModel(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class ExtractiveLLM:
    """Copy the context sentence that best matches the question.

    This is the offline reader. It makes no network calls and only emits text
    that already appears in the retrieved context, or a fixed abstention.
    """

    name = "extractive"

    def generate(self, prompt: str) -> str:
        context, question = _parse_prompt(prompt)
        question_terms = content_terms(question)
        best_sentence = ""
        best_score = 0.0
        for sentence in _candidate_sentences(context):
            score = term_recall(question_terms, content_terms(sentence))
            if score <= 0.0:
                continue
            if score > best_score or (score == best_score and sentence < best_sentence):
                best_score = score
                best_sentence = sentence
        if not best_sentence:
            return ABSTENTION
        return best_sentence


class OpenAIChatLLM:
    name = "openai"

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""


class AnthropicChatLLM:
    name = "anthropic"

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        blocks = []
        for block in response.content:
            text = getattr(block, "text", "")
            if text:
                blocks.append(text)
        return "".join(blocks)


ResolvedLLM = Literal["extractive", "openai", "anthropic"]


def resolve_llm_name(
    requested: LLMName,
    *,
    openai_key: str | None,
    anthropic_key: str | None,
) -> ResolvedLLM:
    if requested == "auto":
        if openai_key:
            return "openai"
        if anthropic_key:
            return "anthropic"
        return "extractive"
    if requested == "extractive":
        return "extractive"
    if requested == "openai":
        if not openai_key:
            raise RuntimeError("RAGQA_LLM=openai requires OPENAI_API_KEY or RAGQA_OPENAI_API_KEY.")
        return "openai"
    if requested == "anthropic":
        if not anthropic_key:
            raise RuntimeError(
                "RAGQA_LLM=anthropic requires ANTHROPIC_API_KEY or RAGQA_ANTHROPIC_API_KEY."
            )
        return "anthropic"
    assert_never(requested)


def build_llm(settings: Settings) -> LanguageModel:
    resolved = resolve_llm_name(
        settings.llm,
        openai_key=openai_api_key(settings),
        anthropic_key=anthropic_api_key(settings),
    )
    if resolved == "extractive":
        return ExtractiveLLM()
    if resolved == "openai":
        return OpenAIChatLLM(OpenAI(api_key=openai_api_key(settings)), model=settings.openai_model)
    if resolved == "anthropic":
        return AnthropicChatLLM(
            Anthropic(api_key=anthropic_api_key(settings)),
            model=settings.anthropic_model,
        )
    assert_never(resolved)


def _candidate_sentences(context: str) -> list[str]:
    candidates = []
    for sentence in sentences(context):
        stripped = _CITATION.sub("", sentence).strip()
        if stripped:
            candidates.append(stripped)
    return candidates


def _parse_prompt(prompt: str) -> tuple[str, str]:
    context_marker = "Context:"
    question_marker = "Question:"
    context_at = prompt.find(context_marker)
    question_at = prompt.find(question_marker)
    if context_at < 0 or question_at < 0 or question_at < context_at:
        return "", prompt.strip()
    context = prompt[context_at + len(context_marker) : question_at].strip()
    question_line = prompt[question_at + len(question_marker) :].strip().splitlines()
    question = question_line[0].strip() if question_line else ""
    return context, question
