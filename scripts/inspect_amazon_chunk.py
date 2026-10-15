#!/usr/bin/env python3
"""Extrae texto de una página del 10-K de Amazon 2017 y genera su embedding.

Extrae la página con índice 37 (página impresa 38) de:
data/FinanceBench/pdfs/AMAZON_2017_10K.pdf

y genera el embedding con Ollama (``nomic-embed-text``).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embeddings.ollama_embeddings import embed_text  # noqa: E402
from ingestion.pdf import extract_page_text  # noqa: E402


PDF_PATH = ROOT / "data" / "FinanceBench" / "pdfs" / "AMAZON_2017_10K.pdf"
DOC_NAME = "AMAZON_2017_10K"
PAGE_INDEX = 37  # índice 0-based; corresponde a la página impresa 38


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF no encontrado: {PDF_PATH}", file=sys.stderr)
        return 1

    try:
        text = extract_page_text(PDF_PATH, PAGE_INDEX)
    except Exception as exc:
        print(f"ERROR al extraer texto del PDF: {exc}", file=sys.stderr)
        return 1

    result = {
        "text": text,
        "doc_name": DOC_NAME,
        "page": PAGE_INDEX + 1,
    }

    print(f"doc_name: {result['doc_name']}")
    print(f"page (printed): {result['page']}")
    print(f"page_index (0-based): {PAGE_INDEX}")
    print(f"longitud_text: {len(text)}")
    print("-" * 60)

    try:
        embedding = embed_text(text)
    except Exception as exc:
        print(f"ERROR al generar embedding: {exc}", file=sys.stderr)
        return 1

    if not embedding:
        print("ERROR: embedding vacío", file=sys.stderr)
        return 1

    print(f"dimensión_embedding: {len(embedding)}")
    print(f"primeros_10_valores: {embedding[:10]}")
    print("-" * 60)
    print(text)
    print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
