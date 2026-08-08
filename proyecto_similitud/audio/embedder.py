"""Generación de embeddings de audio.

Provee una interfaz abstracta AudioEmbedder y una implementación
concreta CLAPEmbedder que usa el modelo laion/clap-htsat-unfused.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Modelo CLAP por defecto
DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000
CLAP_EMBEDDING_DIM = 512


class AudioEmbedder(ABC):
    """Interfaz abstracta para generadores de embeddings de audio.

    Permite intercambiar CLAP por Librosa (o cualquier otro) sin
    modificar las capas superiores (similarity, cli).
    """

    @abstractmethod
    def embed(self, audio_path: Path) -> np.ndarray:
        """Genera un embedding normalizado para un archivo de audio.

        Args:
            audio_path: Path al archivo WAV.

        Returns:
            Vector numpy normalizado (norma L2 = 1.0).
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identificador del modelo usado (para el cache)."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionalidad del embedding generado."""
        ...


class CLAPEmbedder(AudioEmbedder):
    """Embedder basado en CLAP (Contrastive Language-Audio Pretraining).

    Usa el modelo laion/clap-htsat-unfused para generar embeddings
    de 512 dimensiones a partir de audio. Carga lazy del modelo
    (solo al primer uso).
    """

    def __init__(self, model_name: str = DEFAULT_CLAP_MODEL):
        self._model_name = model_name
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Carga el modelo CLAP (lazy, solo al primer uso)."""
        if self._model is not None:
            return

        logger.info(f"Cargando modelo CLAP: {self._model_name}...")
        start = time.time()

        from transformers import ClapModel, ClapProcessor

        self._processor = ClapProcessor.from_pretrained(self._model_name)
        self._model = ClapModel.from_pretrained(self._model_name)
        self._model.eval()  # Modo evaluación (determinista)

        elapsed = time.time() - start
        logger.info(f"Modelo CLAP cargado en {elapsed:.1f}s")

    def embed(self, audio_path: Path) -> np.ndarray:
        """Genera embedding CLAP normalizado para un archivo de audio.

        Args:
            audio_path: Path al archivo WAV (idealmente 48kHz mono).

        Returns:
            Vector numpy de 512 dimensiones, normalizado (L2 = 1.0).
        """
        self._load_model()

        # Cargar audio con librosa (resamplea automáticamente si es necesario)
        logger.info(f"Procesando audio: {audio_path.name}")
        start = time.time()

        waveform, sr = librosa.load(str(audio_path), sr=CLAP_SAMPLE_RATE, mono=True)

        # Preparar inputs para CLAP
        inputs = self._processor(
            audio=waveform,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
        )

        # Inferencia sin gradientes (más eficiente en CPU)
        with torch.no_grad():
            audio_features = self._model.get_audio_features(**inputs)

        # Compatibilidad: versiones recientes de transformers devuelven
        # BaseModelOutputWithPooling en lugar de un tensor directo.
        if hasattr(audio_features, "pooler_output"):
            audio_features = audio_features.pooler_output
        elif hasattr(audio_features, "last_hidden_state"):
            audio_features = audio_features.last_hidden_state[:, 0]

        # Convertir a numpy y normalizar
        embedding = audio_features.squeeze().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        elapsed = time.time() - start
        logger.info(
            f"Embedding generado: {embedding.shape[0]} dims, "
            f"norma={np.linalg.norm(embedding):.4f}, "
            f"tiempo={elapsed:.1f}s"
        )

        return embedding

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return CLAP_EMBEDDING_DIM


class LibrosaEmbedder(AudioEmbedder):
    """Embedder fallback basado en features manuales de Librosa.

    Extrae MFCC, chroma y tempo para construir un vector de ~40 dimensiones.
    Más ligero que CLAP pero menos preciso para similitud perceptual.

    TODO: Implementar en Fase 7 si CLAP resulta demasiado pesado.
    """

    def embed(self, audio_path: Path) -> np.ndarray:
        logger.info(f"Procesando audio con Librosa: {audio_path.name}")

        waveform, sr = librosa.load(str(audio_path), sr=22050, mono=True)

        # MFCC (20 coeficientes, promediados en el tiempo)
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)  # (20,)

        # Chroma (12 bins, promediados)
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # (12,)

        # Spectral contrast (7 bands, promediados)
        contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)  # (7,)

        # Tempo (1 valor)
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        tempo_normalized = np.array([tempo / 200.0])  # normalizar a ~[0, 1]

        # Concatenar todo → vector de 40 dimensiones
        feature_vector = np.concatenate([
            mfcc_mean, chroma_mean, contrast_mean, tempo_normalized
        ])

        # Normalizar L2
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        return feature_vector

    @property
    def model_id(self) -> str:
        return "librosa-manual-v1"

    @property
    def embedding_dim(self) -> int:
        return 40
