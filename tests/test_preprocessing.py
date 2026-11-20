"""Unit-тесты базовой загрузки и разбиения аудио."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_deepfake_detector.src.preprocessing import (
    load_audio,
    split_into_windows,
)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Записать тестовый WAV с float-подтипом без PCM-квантования."""
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def test_load_wav(tmp_path: Path) -> None:
    sample_rate = 16_000
    audio = np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32)
    path = tmp_path / "mono.wav"
    _write_wav(path, audio, sample_rate)

    loaded, loaded_rate = load_audio(path)

    assert loaded_rate == sample_rate
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, audio, atol=1e-6)


def test_stereo_is_converted_to_mono(tmp_path: Path) -> None:
    sample_rate = 16_000
    left = np.full(sample_rate, 0.2, dtype=np.float32)
    right = np.full(sample_rate, 0.6, dtype=np.float32)
    stereo = np.column_stack((left, right))
    path = tmp_path / "stereo.wav"
    _write_wav(path, stereo, sample_rate)

    loaded, _ = load_audio(path)

    assert loaded.ndim == 1
    np.testing.assert_allclose(loaded, 0.4, atol=1e-6)


def test_audio_is_resampled_to_target_rate(tmp_path: Path) -> None:
    source_rate = 8_000
    duration_seconds = 1.25
    time = np.arange(int(source_rate * duration_seconds)) / source_rate
    audio = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    path = tmp_path / "resample.wav"
    _write_wav(path, audio, source_rate)

    loaded, loaded_rate = load_audio(path, target_sample_rate=16_000)

    assert loaded_rate == 16_000
    assert abs(loaded.size - int(16_000 * duration_seconds)) <= 1


def test_split_into_windows_keeps_and_optionally_pads_last_window() -> None:
    audio = np.arange(25, dtype=np.float32)

    windows = split_into_windows(audio, sample_rate=10, window_seconds=1.0)
    padded = split_into_windows(
        audio,
        sample_rate=10,
        window_seconds=1.0,
        pad_last_window=True,
    )

    assert [window.size for window in windows] == [10, 10, 5]
    assert [window.size for window in padded] == [10, 10, 10]
    np.testing.assert_array_equal(padded[-1][:5], audio[-5:])
    np.testing.assert_array_equal(padded[-1][5:], np.zeros(5, dtype=np.float32))


def test_corrupted_wav_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"this is not a wav file")

    with pytest.raises(ValueError, match="Не удалось загрузить аудио"):
        load_audio(path)

