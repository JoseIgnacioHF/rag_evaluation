#!/usr/bin/env python3
"""Retrieval semántico de páginas del 10-K de Amazon 2017.

Recupera las cinco páginas más similares a una pregunta de FinanceBench
usando embeddings de Ollama (``nomic-embed-text``) y similitud coseno.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embeddings.ollama_embeddings import embed_text, embed_texts  # noqa: E402
from ingestion.pdf import extract_pdf_pages  # noqa: E402
from retrieval.similarity import top_k  # noqa: E402


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

    non_empty = [chunk for chunk in chunks if chunk["text"].strip()]
    if not non_empty:
        print("ERROR: ninguna página contiene texto.", file=sys.stderr)
        return 1

    print(f"PDF: {PDF_PATH}")
    print(f"Páginas del PDF: {len(chunks)} (con texto: {len(non_empty)})")
    print(f"Pregunta ({FINANCEBENCH_ID}): {QUESTION}\n")

    try:
        chunk_embeddings = embed_texts([chunk["text"] for chunk in non_empty])
        question_embedding = embed_text(QUESTION)
    except Exception as exc:
        print(f"ERROR generando embeddings: {exc}", file=sys.stderr)
        return 1

    if any(not embedding for embedding in chunk_embeddings) or not question_embedding:
        print("ERROR: se han generado embeddings vacíos.", file=sys.stderr)
        return 1

    question_dim = len(question_embedding)
    for chunk, embedding in zip(non_empty, chunk_embeddings):
        if len(embedding) != question_dim:
            print(
                "ERROR: dimensiones incoherentes entre pregunta y página "
                f"{chunk['page']}: {question_dim} != {len(embedding)}",
                file=sys.stderr,
            )
            return 1

    for chunk, embedding in zip(non_empty, chunk_embeddings):
        chunk["embedding"] = embedding

    results = top_k(question_embedding, non_empty, k=TOP_K)

    print(f"Top {TOP_K} páginas más similares:")
    for rank, (score, chunk) in enumerate(results, start=1):
        print(
            f"{rank}. {chunk['doc_name']} - pág. {chunk['page']} "
            f"- similitud coseno: {score:.4f}"
        )

    best_score, best_chunk = results[0]
    print(
        f"\nPágina mejor clasificada: {best_chunk['doc_name']} "
        f"(pág. {best_chunk['page']}, similitud {best_score:.4f})"
    )
    print("-" * 60)
    print(best_chunk["text"])
    print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
