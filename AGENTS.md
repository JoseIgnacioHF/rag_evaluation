# AGENTS.md

## Environment
- Python 3.10 required (pyproject: `>=3.10,<3.11`). Current venv uses Python 3.10.12.
- Install in editable mode after adding deps: `pip install -e .`. `requirements.txt` is currently empty; add dependencies there or to `pyproject.toml`.
- Keep large artifacts out of git: `data/` is gitignored (`.gitignore`). Don't commit generated vectorstores/results.

## Project layout
- `src/rag_evaluation/`: core package with `ingestion/`, `embeddings/`, `retrieval/`, `generation/`, `models/` (scaffolded). Follow src layout; import as `rag_evaluation.*`.
- `data/`: FinanceBench QA data (`financebench_open_source.jsonl`, `financebench_document_information.jsonl`), PDFs (`pdfs/amazon/`, `FinanceBench/pdfs/`), and reference material.
- `tests/`, `scripts/`, `notebooks/`: placeholders (empty).

## Development notes
- FinanceBench schema: see `data/FinanceBench/README.md` and `data/FinanceBench/evaluation_playground.ipynb`. Join questions (`financebench_open_source.jsonl`) with metadata (`financebench_document_information.jsonl`) on `doc_name`.
- No linter/test/typecheck commands are configured yet. When adding (pytest/ruff/mypy/etc.), record exact commands in this file.

## Gotchas
- `.gitignore` ignores `data/` anywhere; local copies under `data/` won't be committed.
- `requirements.txt` is empty; dependencies aren't installed in `.venv` beyond pip/setuptools.
