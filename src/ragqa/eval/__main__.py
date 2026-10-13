"""Evaluate the bundled golden set: ``python -m ragqa.eval``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragqa.config import Settings
from ragqa.eval.runner import load_golden, render_markdown_table, run_evaluation
from ragqa.service import build_service
from ragqa.types import EmbedderName


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipelines on the golden set.")
    parser.add_argument("--embedder", choices=["hash", "sentence-transformers", "openai"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evals/results.json"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-hit-rate", type=float)
    args = parser.parse_args(argv)

    settings = Settings(auto_ingest=False)
    if args.embedder is not None:
        settings = settings.model_copy(update={"embedder": _embedder(args.embedder)})
    service = build_service(settings)
    try:
        dataset = args.dataset or settings.golden_path
        if service.store.count() == 0:
            service.ingest_directory(settings.corpus_dir)
        report = run_evaluation(service, load_golden(dataset), k=args.k)
    finally:
        service.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(render_markdown_table(report))
    print(f"\nWrote {args.output}")

    if args.min_hit_rate is None:
        return 0
    missed = [
        summary.pipeline
        for summary in report.summaries
        if summary.retrieval_hit_rate < args.min_hit_rate
    ]
    if missed:
        joined = ", ".join(missed)
        print(f"Hit rate below {args.min_hit_rate:.2f} for: {joined}")
        return 1
    return 0


def _embedder(value: str) -> EmbedderName:
    if value == "hash":
        return "hash"
    if value == "sentence-transformers":
        return "sentence-transformers"
    if value == "openai":
        return "openai"
    raise ValueError(value)


if __name__ == "__main__":
    raise SystemExit(main())
