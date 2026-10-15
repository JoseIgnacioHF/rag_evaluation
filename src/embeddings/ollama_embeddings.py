"""Embeddings a través de la API HTTP local de Ollama."""

import json
import urllib.error
import urllib.request

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    """Realiza una petición POST con JSON y devuelve la respuesta decodificada."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTPError llamando a {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con Ollama en {url}: {exc}") from exc


def embed_texts(
    texts: list[str],
    base_url: str = OLLAMA_BASE_URL,
    model: str = EMBED_MODEL,
    timeout: int = 120,
) -> list[list[float]]:
    """Genera embeddings para una lista de textos con el endpoint ``/api/embed``."""
    if not texts:
        raise ValueError("La lista de textos está vacía.")
    payload = {"model": model, "input": texts}
    result = _post_json(f"{base_url}/api/embed", payload, timeout)
    embeddings = result.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Respuesta inesperada de Ollama: {result}")
    return embeddings


def embed_text(
    text: str,
    base_url: str = OLLAMA_BASE_URL,
    model: str = EMBED_MODEL,
    timeout: int = 120,
) -> list[float]:
    """Genera el embedding de un único texto."""
    if not text:
        raise ValueError("El texto está vacío.")
    return embed_texts([text], base_url=base_url, model=model, timeout=timeout)[0]
