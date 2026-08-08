"""Motor de similitud: query ponderada y ranking por similitud coseno.

Construye un vector de consulta como combinación lineal ponderada
de los embeddings de entrada, y rankea los candidatos del catálogo
por similitud coseno.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Pesos por defecto
DEFAULT_WEIGHT_MAIN = 1.0
DEFAULT_WEIGHT_OPTIONAL = 0.4


def build_query_vector(
    main_embeddings: list[np.ndarray],
    optional_embeddings: list[np.ndarray] | None = None,
    weight_main: float = DEFAULT_WEIGHT_MAIN,
    weight_optional: float = DEFAULT_WEIGHT_OPTIONAL,
) -> np.ndarray:
    """Construye un vector de consulta ponderado a partir de los embeddings de entrada.

    Formula: q = normalize( Σ(w_main × e_i) + Σ(w_optional × e_j) )

    Args:
        main_embeddings: Lista de embeddings de las canciones principales (≥1).
        optional_embeddings: Lista de embeddings de canciones opcionales (puede ser vacía/None).
        weight_main: Peso para canciones principales (default: 1.0).
        weight_optional: Peso para canciones opcionales (default: 0.4).

    Returns:
        Vector numpy normalizado (L2 = 1.0) representando la consulta.

    Raises:
        ValueError: Si no se proporcionan embeddings principales.
    """
    if not main_embeddings:
        raise ValueError("Se requiere al menos un embedding principal")

    # Suma ponderada de embeddings principales
    query = np.zeros_like(main_embeddings[0], dtype=np.float64)
    for emb in main_embeddings:
        query += weight_main * emb.astype(np.float64)

    # Suma ponderada de embeddings opcionales
    if optional_embeddings:
        for emb in optional_embeddings:
            query += weight_optional * emb.astype(np.float64)

    # Normalizar L2
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    query = query.astype(np.float32)

    logger.info(
        f"Vector de consulta construido: {len(main_embeddings)} principales "
        f"(w={weight_main}) + {len(optional_embeddings or [])} opcionales "
        f"(w={weight_optional}), norma final={np.linalg.norm(query):.4f}"
    )

    return query


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores.

    Asume que los vectores ya están normalizados (L2 = 1.0),
    por lo que el coseno es simplemente el producto punto.

    Args:
        a: Vector numpy normalizado.
        b: Vector numpy normalizado.

    Returns:
        Similitud coseno ∈ [-1, 1]. Mayor = más similar.
    """
    return float(np.dot(a, b))


def rank_similar(
    query_vector: np.ndarray,
    catalog_embeddings: dict[int, np.ndarray],
    top_n: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Rankea las canciones del catálogo por similitud al vector de consulta.

    Args:
        query_vector: Vector de consulta normalizado.
        catalog_embeddings: Dict de {track_id: embedding} del catálogo.
        top_n: Número de resultados a devolver.
        exclude_ids: Set de track_ids a excluir del ranking (canciones de entrada).

    Returns:
        Lista de (track_id, score) ordenada de mayor a menor similitud.
    """
    if exclude_ids is None:
        exclude_ids = set()

    scores: list[tuple[int, float]] = []

    for track_id, embedding in catalog_embeddings.items():
        if track_id in exclude_ids:
            continue
        score = cosine_similarity(query_vector, embedding)
        scores.append((track_id, score))

    # Ordenar por score descendente
    scores.sort(key=lambda x: x[1], reverse=True)

    # Top-N
    results = scores[:top_n]

    if results:
        logger.info(
            f"Ranking completado: {len(scores)} candidatos evaluados, "
            f"top score={results[0][1]:.4f}, "
            f"bottom score={results[-1][1]:.4f}"
        )
    else:
        logger.warning("Ranking vacío: no hay candidatos en el catálogo")

    return results
