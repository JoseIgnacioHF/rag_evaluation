"""Generación de respuestas con modelos de Ollama.

Construye un prompt a partir de los chunks recuperados y llama a la API HTTP
local de Ollama. No carga PDFs ni ejecuta retrieval.
"""

import json
import urllib.error
import urllib.request
from typing import Any

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "qwen2.5:1.5b"

_SYSTEM_INSTRUCTIONS = (
    "You are a financial analyst assistant. Answer the user's question using "
    "ONLY the information contained in the provided context. If the context "
    "does not contain enough information to answer, say so explicitly instead "
    "of inventing data. Mention the document name and page number you relied on."
)


def build_context(results: list[dict[str, Any]]) -> str:
    """Construye el contexto a partir de los chunks recuperados.

    Cada resultado aporta su texto completo junto con la referencia de
    documento y página para mantener la trazabilidad de las fuentes.
    """
    parts = [
        f"[{result['doc_name']}, page {result['page']}]\n{result['text']}"
        for result in results
    ]
    return "\n\n".join(parts)


def build_prompt(question: str, results: list[dict[str, Any]]) -> str:
    """Construye el prompt final con instrucciones, contexto y pregunta."""
    context = build_context(results)
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer(
    question: str,
    results: list[dict[str, Any]],
    base_url: str = OLLAMA_BASE_URL,
    model: str = GENERATION_MODEL,
    timeout: int = 180,
) -> str:
    """Genera una respuesta a ``question`` usando ``results`` como contexto.

    ``results`` es la salida de ``retrieval.retriever.retrieve``: una lista de
    dicts con las claves ``text``, ``doc_name``, ``page`` y ``score``.
    Devuelve el texto generado por el modelo.
    """
    if not question or not question.strip():
        raise ValueError("La pregunta está vacía.")
    if not results:
        raise ValueError("No hay resultados recuperados para construir el contexto.")

    prompt = build_prompt(question, results)
    payload = {"model": model, "prompt": prompt, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/api/generate"
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTPError llamando a {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con Ollama en {url}: {exc}") from exc

    answer = result.get("response")
    if not answer:
        raise RuntimeError(f"Respuesta inesperada de Ollama: {result}")
    return answer
