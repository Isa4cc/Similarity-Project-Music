"""Descarga y conversión de previews de audio.

Descarga clips de ~30s desde previewUrl (AAC) y los convierte
a WAV 48kHz mono, que es el formato requerido por CLAP.
"""

import logging
import tempfile
from pathlib import Path

import requests
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# CLAP espera audio a 48kHz
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1  # mono


def download_preview(preview_url: str, dest_dir: Path | None = None) -> Path:
    """Descarga un preview de audio y lo convierte a WAV 48kHz mono.

    Args:
        preview_url: URL del preview (típicamente AAC/M4A de iTunes).
        dest_dir: Directorio destino. Si es None, usa un directorio temporal.

    Returns:
        Path al archivo WAV convertido.

    Raises:
        requests.exceptions.RequestException: Si la descarga falla.
        Exception: Si la conversión de audio falla (requiere ffmpeg).
    """
    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="similitud_"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Descargar el archivo de audio
    logger.info(f"Descargando preview: {preview_url}")
    response = requests.get(preview_url, timeout=30)
    response.raise_for_status()

    # Guardar archivo original (AAC/M4A)
    # iTunes sirve .m4a típicamente
    original_path = dest_dir / "preview_original.m4a"
    original_path.write_bytes(response.content)
    logger.debug(
        f"Preview descargado: {len(response.content)} bytes → {original_path}"
    )

    # Convertir a WAV 48kHz mono usando pydub (requiere ffmpeg)
    wav_path = dest_dir / "preview.wav"
    try:
        audio = AudioSegment.from_file(str(original_path))
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        audio = audio.set_channels(TARGET_CHANNELS)
        audio.export(str(wav_path), format="wav")
        logger.info(
            f"Audio convertido a WAV: {TARGET_SAMPLE_RATE}Hz, "
            f"{TARGET_CHANNELS}ch, duración={len(audio)/1000:.1f}s"
        )
    except Exception as e:
        logger.error(
            f"Error al convertir audio. ¿Está ffmpeg instalado? Error: {e}"
        )
        raise

    # Limpiar el archivo original
    original_path.unlink(missing_ok=True)

    return wav_path


def cleanup_preview(wav_path: Path) -> None:
    """Elimina un archivo de preview y su directorio temporal si está vacío.

    Args:
        wav_path: Path al archivo WAV a eliminar.
    """
    try:
        if wav_path.exists():
            wav_path.unlink()
            logger.debug(f"Preview eliminado: {wav_path}")

            # Si el directorio padre quedó vacío, eliminarlo también
            parent = wav_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                logger.debug(f"Directorio temporal eliminado: {parent}")
    except OSError as e:
        logger.warning(f"No se pudo limpiar {wav_path}: {e}")
