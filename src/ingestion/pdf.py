"""Extracción de texto de documentos PDF."""

from pathlib import Path

import pymupdf


def extract_page_text(pdf_path: str | Path, page_index: int) -> str:
    """Extrae el texto de una página concreta (índice 0-based)."""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(
                f"page_index {page_index} fuera de rango (0..{len(doc) - 1})"
            )
        return doc.load_page(page_index).get_text("text") or ""
    finally:
        doc.close()


def extract_pdf_pages(pdf_path: str | Path, doc_name: str) -> list[dict]:
    """Extrae todas las páginas de un PDF.

    Devuelve una lista de chunks (uno por página) con las claves:
    ``text``, ``doc_name`` y ``page`` (numeración 1-based).
    """
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        chunks: list[dict] = []
        for page_index in range(len(doc)):
            text = doc.load_page(page_index).get_text("text") or ""
            chunks.append(
                {
                    "text": text,
                    "doc_name": doc_name,
                    "page": page_index + 1,
                }
            )
        return chunks
    finally:
        doc.close()
