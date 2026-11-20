"""Чистые функции описательного акустического анализа.

Модуль не зависит от Qt и не формирует вердикт REAL/FAKE. Благодаря этому
расчёты можно тестировать и использовать отдельно от desktop-интерфейса.
"""

from typing import TypedDict

import librosa
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class F0Result(TypedDict):
    """Статистика и временной ряд основной частоты."""

    times: FloatArray
    values: FloatArray
    mean: float
    min: float
    max: float
    std: float
    voiced_ratio: float


class MfccResult(TypedDict):
    """Матрица MFCC и покоэффициентная статистика."""

    values: FloatArray
    mean: FloatArray
    std: FloatArray


class SpectralResult(TypedDict):
    """Средние спектральные характеристики записи."""

    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    zero_crossing_rate: float


def _validate_audio(audio: np.ndarray, sample_rate: int) -> None:
    """Проверить общие входные данные акустических функций."""
    if audio.ndim != 1:
        raise ValueError("Ожидается одномерный mono-сигнал")
    if audio.size == 0:
        raise ValueError("Аудиосигнал пуст")
    if sample_rate <= 0:
        raise ValueError("sample_rate должен быть положительным")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Аудиосигнал содержит NaN или бесконечные значения")


def _pad_for_frame_analysis(audio: np.ndarray, minimum_size: int = 2_048) -> np.ndarray:
    """Дополнить очень короткий сигнал нулями для кадровых признаков."""
    if audio.size >= minimum_size:
        return audio
    return np.pad(audio, (0, minimum_size - audio.size))


def calculate_f0(audio: np.ndarray, sample_rate: int) -> F0Result:
    """Рассчитать F0 через pYIN без классификационных порогов."""
    _validate_audio(audio, sample_rate)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sample_rate,
    )
    voiced_f0 = f0[np.isfinite(f0)]
    if voiced_f0.size == 0:
        raise ValueError("Не удалось обнаружить voiced-фрагменты")
    return {
        "times": librosa.times_like(f0, sr=sample_rate),
        "values": f0,
        "mean": float(np.mean(voiced_f0)),
        "min": float(np.min(voiced_f0)),
        "max": float(np.max(voiced_f0)),
        "std": float(np.std(voiced_f0)),
        "voiced_ratio": float(np.mean(voiced_flag)),
    }


def calculate_mfcc(
    audio: np.ndarray, sample_rate: int, coefficient_count: int = 13
) -> MfccResult:
    """Рассчитать MFCC для описательной визуализации.

    ``coefficient_count`` не относится к будущим параметрам CNN или
    Mel-спектрограммы модели.
    """
    _validate_audio(audio, sample_rate)
    if coefficient_count <= 0:
        raise ValueError("coefficient_count должен быть положительным")
    framed_audio = _pad_for_frame_analysis(audio)
    values = librosa.feature.mfcc(
        y=framed_audio,
        sr=sample_rate,
        n_mfcc=coefficient_count,
    )
    return {
        "values": values,
        "mean": np.mean(values, axis=1),
        "std": np.std(values, axis=1),
    }


def calculate_spectral_features(
    audio: np.ndarray, sample_rate: int
) -> SpectralResult:
    """Рассчитать спектральные признаки без определения REAL/FAKE."""
    _validate_audio(audio, sample_rate)
    framed_audio = _pad_for_frame_analysis(audio)
    centroid = librosa.feature.spectral_centroid(y=framed_audio, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=framed_audio, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=framed_audio, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(framed_audio)
    return {
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "zero_crossing_rate": float(np.mean(zcr)),
    }

