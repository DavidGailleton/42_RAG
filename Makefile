PYTHON = uv run python
MAIN = src

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	$(PYTHON) -m $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN)

api:
	uv run fastapi run src/api.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/datasets/*
	rm -rf data/output/search_results/*
	rm -rf data/output/search_results_and_answer/*
	rm -rf data/processed/*

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest
