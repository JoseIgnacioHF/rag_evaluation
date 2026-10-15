#!/usr/bin/env python3
"""Responde una pregunta de FinanceBench con Qwen a partir de contexto recuperado.

Flujo: extrae las páginas del 10-K de Amazon 2017, recupera el top-5 con
``retrieve`` y genera una respuesta con ``qwen2.5:1.5b`` mediante Ollama.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generation.ollama_generator import generate_answer  # noqa: E402
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

    print(f"Pregunta ({FINANCEBENCH_ID}): {QUESTION}\n")

    try:
        results = retrieve(QUESTION, chunks, top_k=TOP_K)
    except Exception as exc:
        print(f"ERROR durante el retrieval: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: el retrieval no devolvió resultados.", file=sys.stderr)
        return 1

    try:
        answer = generate_answer(QUESTION, results)
    except Exception as exc:
        print(f"ERROR durante la generación: {exc}", file=sys.stderr)
        return 1

    print("Respuesta de Qwen:")
    print("-" * 60)
    print(answer)
    print("-" * 60)

    print("\nFuentes recuperadas (top-%d):" % len(results))
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['doc_name']} - pág. {result['page']} "
            f"- similitud coseno: {result['score']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
