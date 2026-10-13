"""Run the golden set through each pipeline and aggregate metrics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ragqa.eval.metrics import (
    answer_correctness,
    answer_relevance,
    context_recall,
    faithfulness,
    retrieval_hit,
)
from ragqa.prompts import format_context
from ragqa.service import QAService
from ragqa.types import GoldenItem, PipelineName

PIPELINES: tuple[PipelineName, ...] = ("langchain", "llamaindex", "tfidf")


@dataclass(frozen=True)
class ExampleScore:
    id: str
    pipeline: str
    faithfulness: float
    answer_relevance: float
    answer_correctness: float
    hit: bool
    context_recall: float
    latency_ms: float
    answer: str
    retrieved_documents: list[str]


@dataclass(frozen=True)
class PipelineSummary:
    pipeline: str
    faithfulness: float
    answer_relevance: float
    answer_correctness: float
    retrieval_hit_rate: float
    context_recall: float
    mean_latency_ms: float
    n: int


@dataclass(frozen=True)
class EvalReport:
    embedder: str
    llm: str
    k: int
    summaries: list[PipelineSummary]
    examples: list[ExampleScore]

    def to_dict(self) -> dict[str, object]:
        return {
            "embedder": self.embedder,
            "llm": self.llm,
            "k": self.k,
            "summaries": [asdict(summary) for summary in self.summaries],
            "examples": [asdict(example) for example in self.examples],
        }


def load_golden(path: Path) -> list[GoldenItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenItem(
            id=item["id"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            document_id=item["document_id"],
            expected_snippets=list(item["expected_snippets"]),
        )
        for item in payload["items"]
    ]


def run_evaluation(
    service: QAService,
    items: list[GoldenItem],
    *,
    k: int = 4,
    pipelines: tuple[PipelineName, ...] = PIPELINES,
) -> EvalReport:
    examples: list[ExampleScore] = []
    for item in items:
        for pipeline in pipelines:
            started = time.perf_counter()
            answer = service.query(item.question, pipeline=pipeline, k=k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            context = format_context(answer.chunks)
            examples.append(
                ExampleScore(
                    id=item.id,
                    pipeline=pipeline,
                    faithfulness=faithfulness(answer.text, context),
                    answer_relevance=answer_relevance(item.question, answer.text),
                    answer_correctness=answer_correctness(item.reference_answer, answer.text),
                    hit=retrieval_hit(item.document_id, answer.chunks),
                    context_recall=context_recall(item.expected_snippets, answer.chunks),
                    latency_ms=elapsed_ms,
                    answer=answer.text,
                    retrieved_documents=[chunk.document_id for chunk in answer.chunks],
                )
            )
    summaries = [_summarize(pipeline, examples) for pipeline in pipelines]
    return EvalReport(
        embedder=service.embedder.name,
        llm=service.llm.name,
        k=k,
        summaries=summaries,
        examples=examples,
    )


def render_markdown_table(report: EvalReport) -> str:
    header = (
        "| Pipeline | Faithfulness | Answer relevance | Answer correctness | "
        "Retrieval hit rate | Context recall | Mean latency (ms) |"
    )
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, divider]
    for summary in report.summaries:
        rows.append(
            "| {pipeline} | {faithfulness:.3f} | {answer_relevance:.3f} | "
            "{answer_correctness:.3f} | {retrieval_hit_rate:.3f} | {context_recall:.3f} | "
            "{mean_latency_ms:.1f} |".format(**asdict(summary))
        )
    return "\n".join(rows)


def _summarize(pipeline: str, examples: list[ExampleScore]) -> PipelineSummary:
    rows = [example for example in examples if example.pipeline == pipeline]
    count = len(rows) or 1
    return PipelineSummary(
        pipeline=pipeline,
        faithfulness=_mean([row.faithfulness for row in rows]),
        answer_relevance=_mean([row.answer_relevance for row in rows]),
        answer_correctness=_mean([row.answer_correctness for row in rows]),
        retrieval_hit_rate=sum(1 for row in rows if row.hit) / count,
        context_recall=_mean([row.context_recall for row in rows]),
        mean_latency_ms=_mean([row.latency_ms for row in rows]),
        n=len(rows),
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
