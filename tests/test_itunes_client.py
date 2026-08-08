"""Tests para el cliente de iTunes Search API."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from proyecto_similitud.data.itunes_client import (
    find_best_match,
    search_tracks,
    _MIN_REQUEST_INTERVAL,
)


# Respuesta mockeada de iTunes
MOCK_ITUNES_RESPONSE = {
    "resultCount": 2,
    "results": [
        {
            "trackId": 1440833237,
            "trackName": "Karma Police",
            "artistName": "Radiohead",
            "collectionName": "OK Computer",
            "previewUrl": "https://audio-ssl.itunes.apple.com/preview/karma_police.m4a",
            "trackTimeMillis": 263000,
        },
        {
            "trackId": 1440833238,
            "trackName": "Karma Police (Live)",
            "artistName": "Radiohead",
            "collectionName": "I Might Be Wrong",
            "previewUrl": None,
            "trackTimeMillis": 300000,
        },
    ],
}


@patch("proyecto_similitud.data.itunes_client.time.sleep")
@patch("proyecto_similitud.data.itunes_client.requests.get")
def test_search_tracks_returns_results(mock_get, mock_sleep):
    """Verifica que search_tracks parsea correctamente la respuesta de iTunes."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_ITUNES_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    results = search_tracks("Karma Police", limit=5)

    assert len(results) == 2
    assert results[0]["trackName"] == "Karma Police"
    assert results[0]["trackId"] == 1440833237


@patch("proyecto_similitud.data.itunes_client.time.sleep")
@patch("proyecto_similitud.data.itunes_client.requests.get")
def test_search_tracks_empty_results(mock_get, mock_sleep):
    """Verifica comportamiento cuando iTunes no devuelve resultados."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"resultCount": 0, "results": []}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    results = search_tracks("cancion_inexistente_xyz")

    assert results == []


@patch("proyecto_similitud.data.itunes_client.time.sleep")
@patch("proyecto_similitud.data.itunes_client.requests.get")
def test_search_tracks_retry_on_429(mock_get, mock_sleep):
    """Verifica que se reintenta en caso de rate limit (429)."""
    # Primer intento: 429, segundo intento: éxito
    error_response = MagicMock()
    error_response.status_code = 429
    error_response.url = "https://itunes.apple.com/search"

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = MOCK_ITUNES_RESPONSE
    success_response.raise_for_status.return_value = None

    http_error = requests.exceptions.HTTPError(response=error_response)
    mock_get.side_effect = [
        MagicMock(raise_for_status=MagicMock(side_effect=http_error)),
        success_response,
    ]

    # Debería funcionar en el segundo intento
    results = search_tracks("Karma Police")
    assert len(results) == 2


@patch("proyecto_similitud.data.itunes_client.time.sleep")
@patch("proyecto_similitud.data.itunes_client.requests.get")
def test_find_best_match_prioritizes_preview(mock_get, mock_sleep):
    """Verifica que find_best_match devuelve el primer resultado con previewUrl."""
    # Invertir el orden: el primero NO tiene preview
    inverted_results = {
        "resultCount": 2,
        "results": [
            MOCK_ITUNES_RESPONSE["results"][1],  # sin previewUrl
            MOCK_ITUNES_RESPONSE["results"][0],  # con previewUrl
        ],
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = inverted_results
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = find_best_match("Karma Police")

    # Debe devolver el que tiene previewUrl, no el primero
    assert result is not None
    assert result["trackName"] == "Karma Police"
    assert result["previewUrl"] is not None


@patch("proyecto_similitud.data.itunes_client.time.sleep")
@patch("proyecto_similitud.data.itunes_client.requests.get")
def test_find_best_match_returns_none_on_empty(mock_get, mock_sleep):
    """Verifica que find_best_match devuelve None si no hay resultados."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"resultCount": 0, "results": []}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = find_best_match("cancion_que_no_existe_xyz")

    assert result is None
