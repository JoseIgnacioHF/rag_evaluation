#!/usr/bin/env python3
"""Demostración de retrieval semántico sobre el 10-K de Amazon 2017.

Recupera las cinco páginas más similares a una pregunta de FinanceBench
usando la función pública ``retrieval.retriever.retrieve``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion.pdf import extract_pdf_pages  # noqa: E402
from retrieval.retriever import retrieve  # noqa: E402


PDF_PATH = ROOT / "data" / "FinanceBench" / "pdfs" / "AMAZON_2017_10K.pdf"
DOC_NAME = "AMAZON_2017_10K"
QUESTION = (
    "What is Amazon's year-over-year change in revenue from FY2016 to FY2017? "
    "Calculate what was asked by utilizing the line items clearly shown in the "
    "statement of income."
)
FINANCEBENCH_ID = "financebench_id_08135"
TOP_K = 5


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF no encontrado: {PDF_PATH}", file=sys.stderr)
        return 1

    chunks = extract_pdf_pages(PDF_PATH, DOC_NAME)
    if not chunks:
        print("ERROR: no se extrajo ninguna página del PDF.", file=sys.stderr)
        return 1

    with_text = sum(1 for chunk in chunks if chunk["text"].strip())

    print(f"PDF: {PDF_PATH}")
    print(f"Páginas del PDF: {len(chunks)} (con texto: {with_text})")
    print(f"Pregunta ({FINANCEBENCH_ID}): {QUESTION}\n")

    try:
        results = retrieve(QUESTION, chunks, top_k=TOP_K)
    except Exception as exc:
        print(f"ERROR durante el retrieval: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: el retrieval no devolvió resultados.", file=sys.stderr)
        return 1

    print(f"Top {len(results)} páginas más similares:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['doc_name']} - pág. {result['page']} "
            f"- similitud coseno: {result['score']:.4f}"
        )

    best = results[0]
    print(
        f"\nPágina mejor clasificada: {best['doc_name']} "
        f"(pág. {best['page']}, similitud {best['score']:.4f})"
    )
    print("-" * 60)
    print(best["text"])
    print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
