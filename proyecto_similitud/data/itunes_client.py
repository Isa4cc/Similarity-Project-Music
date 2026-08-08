"""Cliente para la iTunes Search API.

Busca canciones por término y devuelve metadata incluyendo previewUrl.
Implementa throttling (20 req/min) y retry con backoff exponencial.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

# iTunes permite ~20 requests/min por IP
_MIN_REQUEST_INTERVAL = 3.0  # segundos entre requests
_last_request_time: float = 0.0

# Retry config
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # segundos


def _throttle() -> None:
    """Espera lo necesario para respetar el rate limit de iTunes."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        wait = _MIN_REQUEST_INTERVAL - elapsed
        logger.debug(f"Throttling: esperando {wait:.1f}s antes del siguiente request")
        time.sleep(wait)
    _last_request_time = time.time()


def search_tracks(
    query: str,
    country: str = "US",
    limit: int = 10,
    media: str = "music",
    entity: str = "song",
) -> list[dict[str, Any]]:
    """Busca canciones en iTunes Search API.

    Args:
        query: Término de búsqueda (nombre de canción, artista, etc.)
        country: Código de país ISO 3166-1 alpha-2 (default: US)
        limit: Número máximo de resultados (default: 10, max: 200)
        media: Tipo de media a buscar (default: music)
        entity: Tipo de entidad (default: song)

    Returns:
        Lista de diccionarios con metadata de cada canción.
        Cada dict contiene al menos: trackId, trackName, artistName,
        collectionName, previewUrl (si disponible).

    Raises:
        requests.exceptions.RequestException: Si todos los reintentos fallan.
    """
    params = {
        "term": query,
        "country": country,
        "limit": min(limit, 200),
        "media": media,
        "entity": entity,
    }

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        _throttle()
        try:
            logger.info(
                f"Buscando en iTunes: '{query}' (intento {attempt}/{_MAX_RETRIES})"
            )
            response = requests.get(
                ITUNES_SEARCH_URL, params=params, timeout=15
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            logger.info(f"iTunes devolvió {len(results)} resultados para '{query}'")
            return results

        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else "?"
            if status == 429:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    f"Rate limit alcanzado (429). Esperando {wait:.0f}s..."
                )
                time.sleep(wait)
            elif status == 404:
                logger.error(f"Endpoint no encontrado (404). URL: {e.response.url}")
                break  # No tiene sentido reintentar un 404
            else:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(f"Error HTTP {status}. Reintentando en {wait:.0f}s...")
                time.sleep(wait)

        except requests.exceptions.ConnectionError as e:
            last_error = e
            wait = _BACKOFF_BASE ** attempt
            logger.warning(f"Error de conexión. Reintentando en {wait:.0f}s...")
            time.sleep(wait)

        except requests.exceptions.Timeout as e:
            last_error = e
            wait = _BACKOFF_BASE ** attempt
            logger.warning(f"Timeout. Reintentando en {wait:.0f}s...")
            time.sleep(wait)

    logger.error(f"Todos los intentos fallaron para '{query}': {last_error}")
    raise requests.exceptions.RequestException(
        f"No se pudo completar la búsqueda de '{query}' después de "
        f"{_MAX_RETRIES} intentos: {last_error}"
    )


def find_best_match(
    query: str, country: str = "US"
) -> dict[str, Any] | None:
    """Busca una canción y devuelve el mejor match (primer resultado con preview).

    Args:
        query: Nombre de la canción (opcionalmente con artista)
        country: Código de país

    Returns:
        Dict con metadata de la canción, o None si no se encuentra.
    """
    results = search_tracks(query, country=country, limit=5)

    # Priorizar resultados que tengan previewUrl
    for track in results:
        if track.get("previewUrl"):
            logger.info(
                f"Match encontrado: '{track['trackName']}' "
                f"por {track['artistName']}"
            )
            return track

    # Si ninguno tiene preview, devolver el primero (si existe)
    if results:
        logger.warning(
            f"Match encontrado para '{query}' pero sin previewUrl: "
            f"'{results[0].get('trackName')}'"
        )
        return results[0]

    logger.warning(f"No se encontraron resultados para '{query}'")
    return None
