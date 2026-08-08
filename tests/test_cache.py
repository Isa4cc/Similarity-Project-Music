"""Tests para el cache de embeddings en SQLite."""

import numpy as np
import pytest

from proyecto_similitud.storage.cache import EmbeddingCache


@pytest.fixture
def cache():
    """Cache en memoria para tests (no toca disco)."""
    c = EmbeddingCache(":memory:")
    yield c
    c.close()


@pytest.fixture
def sample_embedding():
    """Embedding de ejemplo (512 dims, normalizado)."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def test_put_and_get(cache, sample_embedding):
    """Verifica roundtrip: guardar → recuperar → comparar."""
    cache.put(
        track_id=12345,
        track_name="Karma Police",
        artist_name="Radiohead",
        embedding=sample_embedding,
        album_name="OK Computer",
        preview_url="https://example.com/preview.m4a",
    )

    recovered = cache.get(12345)
    assert recovered is not None
    np.testing.assert_array_almost_equal(recovered, sample_embedding, decimal=6)


def test_get_nonexistent(cache):
    """Verifica que get devuelve None para track_id inexistente."""
    assert cache.get(99999) is None


def test_has(cache, sample_embedding):
    """Verifica el método has."""
    assert not cache.has(12345)
    cache.put(12345, "Test", "Test Artist", sample_embedding)
    assert cache.has(12345)


def test_count(cache, sample_embedding):
    """Verifica el conteo de embeddings."""
    assert cache.count() == 0

    cache.put(1, "Song 1", "Artist 1", sample_embedding)
    assert cache.count() == 1

    cache.put(2, "Song 2", "Artist 2", sample_embedding)
    assert cache.count() == 2


def test_get_all(cache):
    """Verifica que get_all devuelve todos los embeddings."""
    rng = np.random.default_rng(42)

    embeddings = {}
    for i in range(5):
        vec = rng.standard_normal(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        embeddings[i] = vec
        cache.put(i, f"Song {i}", f"Artist {i}", vec)

    all_embs = cache.get_all()

    assert len(all_embs) == 5
    for track_id, emb in embeddings.items():
        np.testing.assert_array_almost_equal(all_embs[track_id], emb, decimal=6)


def test_list_tracks(cache, sample_embedding):
    """Verifica que list_tracks devuelve metadata sin embeddings."""
    cache.put(1, "Karma Police", "Radiohead", sample_embedding, album_name="OK Computer")
    cache.put(2, "Creep", "Radiohead", sample_embedding, album_name="Pablo Honey")

    tracks = cache.list_tracks()

    assert len(tracks) == 2
    assert all("track_name" in t for t in tracks)
    assert all("embedding" not in t for t in tracks)


def test_get_track_metadata(cache, sample_embedding):
    """Verifica que get_track_metadata devuelve los campos correctos."""
    cache.put(
        track_id=999,
        track_name="Lucky",
        artist_name="Radiohead",
        embedding=sample_embedding,
        album_name="OK Computer",
        preview_url="https://example.com/lucky.m4a",
    )

    meta = cache.get_track_metadata(999)
    assert meta is not None
    assert meta["track_name"] == "Lucky"
    assert meta["artist_name"] == "Radiohead"
    assert meta["album_name"] == "OK Computer"
    assert meta["preview_url"] == "https://example.com/lucky.m4a"


def test_put_replaces_existing(cache, sample_embedding):
    """Verifica que put con el mismo track_id reemplaza el embedding."""
    cache.put(1, "Original", "Artist", sample_embedding)

    new_emb = np.ones(512, dtype=np.float32)
    new_emb /= np.linalg.norm(new_emb)
    cache.put(1, "Updated", "Artist", new_emb)

    assert cache.count() == 1
    meta = cache.get_track_metadata(1)
    assert meta["track_name"] == "Updated"

    recovered = cache.get(1)
    np.testing.assert_array_almost_equal(recovered, new_emb, decimal=6)


def test_context_manager():
    """Verifica que el context manager funciona correctamente."""
    with EmbeddingCache(":memory:") as cache:
        assert cache.count() == 0
    # No debería lanzar excepción al salir
