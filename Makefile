.PHONY: test eval serve install

install:
	pip install -e ".[dev,embeddings]"

test:
	pytest

eval:
	python -m ragqa.eval --embedder hash --output evals/results.json --min-hit-rate 1.0

serve:
	uvicorn ragqa.api.app:create_app --factory --host 0.0.0.0 --port 8000
