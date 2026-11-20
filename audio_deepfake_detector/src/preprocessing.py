"""Базовая предобработка аудио без извлечения Mel-спектрограмм."""

from pathlib import Path
from math import gcd

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


def load_audio(
    audio_path: str | Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Загрузить аудио, преобразовать его в mono и привести к нужной частоте.

    Args:
        audio_path: Путь к аудиофайлу, который поддерживает libsndfile.
        target_sample_rate: Целевая частота дискретизации в Гц.

    Returns:
        Кортеж из одномерного массива ``float32`` и частоты дискретизации.
    """
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate должен быть положительным")

    try:
        audio, original_sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=True
        )
    except Exception as soundfile_error:
        # Fallback помогает читать MP3/OGG в окружениях, где libsndfile
        # не поддерживает конкретный codec, но его поддерживает librosa.
        try:
            fallback_audio, original_sample_rate = librosa.load(
                str(audio_path), sr=None, mono=False
            )
            if fallback_audio.ndim == 1:
                audio = fallback_audio[:, np.newaxis]
            else:
                audio = fallback_audio.T
        except Exception as librosa_error:
            raise ValueError(
                f"Не удалось загрузить аудио: {soundfile_error}; {librosa_error}"
            ) from librosa_error

    if audio.size == 0:
        raise ValueError(f"Аудиофайл пуст: {audio_path}")

    # soundfile возвращает форму (число отсчётов, число каналов).
    mono_audio = audio.mean(axis=1)

    if original_sample_rate != target_sample_rate:
        common_divisor = gcd(original_sample_rate, target_sample_rate)
        mono_audio = resample_poly(
            mono_audio,
            up=target_sample_rate // common_divisor,
            down=original_sample_rate // common_divisor,
        ).astype(np.float32)

    return mono_audio.astype(np.float32, copy=False), target_sample_rate


def split_into_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    pad_last_window: bool = False,
) -> list[np.ndarray]:
    """Разбить одномерный сигнал на неперекрывающиеся окна.

    Последнее короткое окно сохраняется. Если ``pad_last_window=True``, оно
    дополняется нулями до полной длины окна.
    """
    if audio.ndim != 1:
        raise ValueError("audio должен быть одномерным mono-сигналом")
    if sample_rate <= 0:
        raise ValueError("sample_rate должен быть положительным")
    if window_seconds <= 0:
        raise ValueError("window_seconds должен быть положительным")

    window_size = int(round(sample_rate * window_seconds))
    if window_size < 1:
        raise ValueError("Размер окна получился меньше одного отсчёта")
    if audio.size == 0:
        return []

    windows = [
        audio[start : start + window_size].copy()
        for start in range(0, audio.size, window_size)
    ]

    if pad_last_window and windows[-1].size < window_size:
        windows[-1] = np.pad(
            windows[-1], (0, window_size - windows[-1].size)
        ).astype(np.float32)

    return windows


def load_and_split_audio(
    audio_path: str | Path,
    window_seconds: float,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    pad_last_window: bool = False,
) -> tuple[list[np.ndarray], int]:
    """Загрузить аудио и сразу разбить его на окна заданной длительности."""
    audio, sample_rate = load_audio(audio_path, target_sample_rate)
    windows = split_into_windows(
        audio,
        sample_rate,
        window_seconds,
        pad_last_window=pad_last_window,
    )
    return windows, sample_rate

