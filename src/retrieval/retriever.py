"""Retrieval semántico de chunks.

Coordina la generación de embeddings con Ollama y el cálculo de similitud
coseno para devolver los chunks más relevantes a una consulta.
"""

from typing import Any

from embeddings.ollama_embeddings import embed_text, embed_texts
from retrieval.similarity import top_k as rank_by_similarity

_REQUIRED_KEYS = ("text", "doc_name", "page")


def retrieve(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Devuelve los ``top_k`` chunks más similares a ``query``.

    Cada chunk de entrada debe contener las claves ``text``, ``doc_name`` y
    ``page``. Los chunks sin texto se ignoran.

    Devuelve una lista de copias de los chunks recuperados, ordenadas de mayor
    a menor similitud coseno, añadiendo la clave ``score``. No imprime nada.
    """
    if not query or not query.strip():
        raise ValueError("La consulta está vacía.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(f"top_k inválido: {top_k!r} (debe ser un entero > 0)")

    candidates: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        missing = [key for key in _REQUIRED_KEYS if key not in chunk]
        if missing:
            raise ValueError(f"Chunk {index} sin claves requeridas: {missing}")
        if chunk["text"] and chunk["text"].strip():
            candidates.append(dict(chunk))

    if not candidates:
        raise ValueError("No hay chunks con texto para recuperar.")

    query_embedding = embed_text(query)
    chunk_embeddings = embed_texts([chunk["text"] for chunk in candidates])

    if not query_embedding or any(not emb for emb in chunk_embeddings):
        raise ValueError("Se han generado embeddings vacíos.")

    query_dim = len(query_embedding)
    for chunk, embedding in zip(candidates, chunk_embeddings):
        if len(embedding) != query_dim:
            raise ValueError(
                "Dimensiones incoherentes entre la consulta y el chunk "
                f"({chunk['doc_name']}, pág. {chunk['page']}): "
                f"{query_dim} != {len(embedding)}"
            )
        chunk["embedding"] = embedding

    ranked = rank_by_similarity(query_embedding, candidates, k=top_k)

    results: list[dict[str, Any]] = []
    for score, chunk in ranked:
        results.append(
            {
                "text": chunk["text"],
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
                "score": score,
            }
        )
    return results
