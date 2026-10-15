#!/usr/bin/env python3
"""Extract text from a specific page of an Amazon 10-K PDF and embed with Ollama.

Extracts page index 37 (printed page 38) from:
data/FinanceBench/pdfs/AMAZON_2017_10K.pdf

Embeds the extracted text using Ollama API and nomic-embed-text.
"""

from pathlib import Path
import sys
import json
import urllib.request
import urllib.error

import pymupdf


OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str, base_url: str = OLLAMA_BASE_URL, model: str = EMBED_MODEL) -> list[float]:
    """Generate embedding for text using Ollama /api/embeddings or /api/embed.

    Tries /api/embeddings first (common), falls back to /api/embed if needed.
    """
    if not text:
        raise ValueError("text vacío, no se puede generar embedding")

    payload_emb = {
        "model": model,
        "prompt": text,
    }
    data_emb = json.dumps(payload_emb).encode("utf-8")
    req_emb = urllib.request.Request(
        f"{base_url}/api/embeddings",
        data=data_emb,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req_emb, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if "embedding" in res:
                return res["embedding"]
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        raise RuntimeError(f"HTTPError al llamar /api/embeddings: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError al conectar con Ollama ({base_url}): {e}") from e

    # Fallback to /api/embed
    payload_e = {
        "model": model,
        "input": text,
    }
    data_e = json.dumps(payload_e).encode("utf-8")
    req_e = urllib.request.Request(
        f"{base_url}/api/embed",
        data=data_e,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req_e, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if "embeddings" in res and res["embeddings"]:
                return res["embeddings"][0]
            if "embedding" in res:
                return res["embedding"]
            raise RuntimeError(f"respuesta inesperada de /api/embed: {res}")
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        raise RuntimeError(f"HTTPError al llamar /api/embed: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError al conectar con Ollama ({base_url}): {e}") from e


def main() -> int:
    root = Path.cwd()
    pdf_path = root / "data" / "FinanceBench" / "pdfs" / "AMAZON_2017_10K.pdf"

    if not pdf_path.exists():
        print(f"ERROR: PDF no encontrado: {pdf_path}", file=sys.stderr)
        return 1

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"ERROR: no se pudo abrir el PDF: {e}", file=sys.stderr)
        return 1

    page_index = 37  # printed page 38
    if page_index < 0 or page_index >= len(doc):
        print(f"ERROR: índice {page_index} fuera de rango (0..{len(doc)-1})", file=sys.stderr)
        doc.close()
        return 1

    page = doc.load_page(page_index)
    text = page.get_text("text") or ""

    result = {
        "text": text,
        "doc_name": "AMAZON_2017_10K",
        "page": 38,
    }

    print(f"doc_name: {result['doc_name']}")
    print(f"page (printed): {result['page']}")
    print(f"page_index (0-based): {page_index}")
    print(f"longitud_text: {len(text)}")
    print("-" * 60)

    try:
        embedding = embed_text(text, base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
    except Exception as e:
        print(f"ERROR al generar embedding: {e}", file=sys.stderr)
        doc.close()
        return 1

    if not embedding:
        print("ERROR: embedding vacío", file=sys.stderr)
        doc.close()
        return 1

    print(f"dimensión_embedding: {len(embedding)}")
    print(f"primeros_10_valores: {embedding[:10]}")
    print("-" * 60)
    print(text)
    print("-" * 60)

    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
