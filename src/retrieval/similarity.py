"""Similitud coseno y recuperación de los mejores resultados."""

import math
from typing import Any


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores.

    Devuelve ``0.0`` si alguno de los vectores tiene norma cero para evitar
    divisiones por cero.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Dimensiones distintas: {len(vector_a)} != {len(vector_b)}"
        )
    dot = sum(x * y for x, y in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(x * x for x in vector_a))
    norm_b = math.sqrt(sum(y * y for y in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_k(
    query_vector: list[float],
    items: list[dict[str, Any]],
    k: int = 5,
    embedding_key: str = "embedding",
) -> list[tuple[float, dict[str, Any]]]:
    """Devuelve los ``k`` elementos más similares al vector de consulta.

    Cada elemento es una tupla ``(similitud, item)`` ordenada de mayor a menor.
    """
    scored = [
        (cosine_similarity(query_vector, item[embedding_key]), item)
        for item in items
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]
