"""Tests para el motor de similitud."""

import numpy as np
import pytest

from proyecto_similitud.similarity.engine import (
    build_query_vector,
    cosine_similarity,
    rank_similar,
)


class TestCoseSimilarity:
    """Tests para la función de similitud coseno."""

    def test_identical_vectors(self):
        """Vectores idénticos → similitud = 1.0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Vectores ortogonales → similitud = 0.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Vectores opuestos → similitud = -1.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_normalized_vectors(self):
        """Similitud coseno con vectores normalizados."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(512).astype(np.float32)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(512).astype(np.float32)
        b /= np.linalg.norm(b)

        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


class TestBuildQueryVector:
    """Tests para la construcción del vector de consulta."""

    def test_single_main_embedding(self):
        """Un solo embedding principal → query = embedding normalizado."""
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        vec /= np.linalg.norm(vec)

        query = build_query_vector([vec])

        np.testing.assert_array_almost_equal(query, vec, decimal=5)

    def test_query_is_normalized(self):
        """El vector de consulta siempre tiene norma L2 ≈ 1.0."""
        rng = np.random.default_rng(42)
        mains = [rng.standard_normal(512).astype(np.float32) for _ in range(5)]
        opts = [rng.standard_normal(512).astype(np.float32) for _ in range(3)]

        query = build_query_vector(mains, opts)

        assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-5)

    def test_optional_has_less_weight(self):
        """Los embeddings opcionales influyen menos que los principales."""
        # Dos vectores ortogonales
        main_vec = np.zeros(512, dtype=np.float32)
        main_vec[0] = 1.0  # apunta a dimensión 0

        opt_vec = np.zeros(512, dtype=np.float32)
        opt_vec[1] = 1.0  # apunta a dimensión 1

        query = build_query_vector([main_vec], [opt_vec], weight_main=1.0, weight_optional=0.4)

        # La componente 0 (main) debe dominar sobre la componente 1 (optional)
        assert query[0] > query[1]

    def test_raises_on_empty_main(self):
        """Error si no hay embeddings principales."""
        with pytest.raises(ValueError, match="al menos un embedding"):
            build_query_vector([])

    def test_no_optionals(self):
        """Funciona correctamente sin embeddings opcionales."""
        rng = np.random.default_rng(42)
        mains = [rng.standard_normal(512).astype(np.float32) for _ in range(3)]

        query = build_query_vector(mains, optional_embeddings=None)

        assert query.shape == (512,)
        assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-5)


class TestRankSimilar:
    """Tests para el ranking de canciones similares."""

    def test_ranking_order(self):
        """Verifica que el ranking ordena por similitud descendente."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        catalog = {
            1: np.array([0.9, 0.1, 0.0], dtype=np.float32),  # más similar
            2: np.array([0.5, 0.5, 0.0], dtype=np.float32),  # medio
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # ortogonal
        }
        # Normalizar
        for k in catalog:
            catalog[k] /= np.linalg.norm(catalog[k])

        results = rank_similar(query, catalog, top_n=3)

        assert len(results) == 3
        assert results[0][0] == 1  # track_id más similar primero
        assert results[0][1] > results[1][1] > results[2][1]  # scores descendentes

    def test_top_n_limits_results(self):
        """Verifica que top_n limita el número de resultados."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        catalog = {i: np.random.randn(2).astype(np.float32) for i in range(100)}

        results = rank_similar(query, catalog, top_n=5)

        assert len(results) == 5

    def test_exclude_ids(self):
        """Verifica que las canciones de entrada se excluyen del ranking."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        catalog = {
            1: np.array([1.0, 0.0], dtype=np.float32),  # idéntico al query
            2: np.array([0.9, 0.1], dtype=np.float32),
            3: np.array([0.5, 0.5], dtype=np.float32),
        }

        results = rank_similar(query, catalog, top_n=10, exclude_ids={1})

        track_ids = [r[0] for r in results]
        assert 1 not in track_ids
        assert 2 in track_ids

    def test_empty_catalog(self):
        """Ranking con catálogo vacío devuelve lista vacía."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = rank_similar(query, {}, top_n=10)
        assert results == []
