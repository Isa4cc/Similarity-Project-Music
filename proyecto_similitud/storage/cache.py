"""Cache de embeddings en SQLite.

Almacena embeddings de audio indexados por trackId de iTunes,
evitando recomputar embeddings ya calculados.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/embeddings.db")


class EmbeddingCache:
    """Cache de embeddings usando SQLite como almacenamiento.

    Cada embedding se guarda como BLOB (bytes crudos del numpy array).
    La metadata de la canción se guarda junto al embedding para
    poder mostrar resultados sin queries adicionales a iTunes.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        """Inicializa el cache, creando la DB y tabla si no existen.

        Args:
            db_path: Ruta al archivo SQLite. Se crea si no existe.
                     Usar ':memory:' para tests.
        """
        self._db_path = Path(db_path) if not isinstance(db_path, str) or db_path != ":memory:" else db_path

        if isinstance(self._db_path, Path):
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path) if isinstance(self._db_path, Path) else self._db_path
        )
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        logger.info(f"Cache de embeddings abierto: {self._db_path}")

    def _create_table(self) -> None:
        """Crea la tabla de embeddings si no existe."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                track_id    INTEGER PRIMARY KEY,
                track_name  TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                album_name  TEXT,
                preview_url TEXT,
                embedding   BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                model_id    TEXT DEFAULT 'clap-htsat-unfused',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def get(self, track_id: int) -> np.ndarray | None:
        """Recupera un embedding del cache.

        Args:
            track_id: iTunes trackId.

        Returns:
            Vector numpy del embedding, o None si no existe.
        """
        row = self._conn.execute(
            "SELECT embedding, embedding_dim FROM embeddings WHERE track_id = ?",
            (track_id,),
        ).fetchone()

        if row is None:
            return None

        embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        expected_dim = row["embedding_dim"]
        if embedding.shape[0] != expected_dim:
            logger.warning(
                f"Embedding para track_id={track_id} tiene dimensión "
                f"{embedding.shape[0]}, se esperaba {expected_dim}. "
                f"Podría estar corrupto."
            )
        return embedding

    def put(
        self,
        track_id: int,
        track_name: str,
        artist_name: str,
        embedding: np.ndarray,
        album_name: str | None = None,
        preview_url: str | None = None,
        model_id: str = "clap-htsat-unfused",
    ) -> None:
        """Guarda o actualiza un embedding en el cache.

        Args:
            track_id: iTunes trackId (clave primaria).
            track_name: Nombre de la canción.
            artist_name: Nombre del artista.
            embedding: Vector numpy del embedding.
            album_name: Nombre del álbum (opcional).
            preview_url: URL del preview (opcional).
            model_id: Identificador del modelo usado.
        """
        embedding_blob = embedding.astype(np.float32).tobytes()
        embedding_dim = embedding.shape[0]

        self._conn.execute(
            """
            INSERT OR REPLACE INTO embeddings
                (track_id, track_name, artist_name, album_name, preview_url,
                 embedding, embedding_dim, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track_id,
                track_name,
                artist_name,
                album_name,
                preview_url,
                embedding_blob,
                embedding_dim,
                model_id,
            ),
        )
        self._conn.commit()
        logger.debug(
            f"Embedding guardado: track_id={track_id}, "
            f"'{track_name}' por {artist_name} ({embedding_dim} dims)"
        )

    def get_all(self) -> dict[int, np.ndarray]:
        """Recupera todos los embeddings del cache.

        Returns:
            Dict de {track_id: embedding_array}.
        """
        rows = self._conn.execute(
            "SELECT track_id, embedding, embedding_dim FROM embeddings"
        ).fetchall()

        result = {}
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            result[row["track_id"]] = emb

        logger.debug(f"Recuperados {len(result)} embeddings del cache")
        return result

    def get_track_metadata(self, track_id: int) -> dict[str, Any] | None:
        """Recupera la metadata de una canción (sin el embedding).

        Args:
            track_id: iTunes trackId.

        Returns:
            Dict con track_name, artist_name, album_name, preview_url.
        """
        row = self._conn.execute(
            """
            SELECT track_id, track_name, artist_name, album_name, preview_url,
                   model_id, created_at
            FROM embeddings WHERE track_id = ?
            """,
            (track_id,),
        ).fetchone()

        if row is None:
            return None
        return dict(row)

    def list_tracks(self) -> list[dict[str, Any]]:
        """Lista todas las canciones en el cache (sin embeddings).

        Returns:
            Lista de dicts con metadata de cada canción.
        """
        rows = self._conn.execute(
            """
            SELECT track_id, track_name, artist_name, album_name, model_id,
                   created_at
            FROM embeddings
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """Cuenta el número de embeddings en el cache."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM embeddings").fetchone()
        return row["cnt"]

    def has(self, track_id: int) -> bool:
        """Verifica si un embedding existe en el cache."""
        row = self._conn.execute(
            "SELECT 1 FROM embeddings WHERE track_id = ?", (track_id,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        """Cierra la conexión a la base de datos."""
        self._conn.close()
        logger.debug("Cache de embeddings cerrado")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
