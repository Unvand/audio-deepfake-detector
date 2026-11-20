"""Unit-тесты независимых акустических расчётов."""

import numpy as np

from audio_deepfake_detector.src.audio_analysis import (
    calculate_f0,
    calculate_mfcc,
    calculate_spectral_features,
)

SAMPLE_RATE = 16_000


def _sine(frequency: float = 220.0, duration: float = 2.0) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_f0_for_sine_wave_is_close_to_known_frequency() -> None:
    result = calculate_f0(_sine(220.0), SAMPLE_RATE)

    assert abs(result["mean"] - 220.0) < 5.0
    assert 0.0 <= result["voiced_ratio"] <= 1.0
    assert result["values"].shape == result["times"].shape


def test_mfcc_shape_and_statistics() -> None:
    result = calculate_mfcc(_sine(), SAMPLE_RATE)

    assert result["values"].shape[0] == 13
    assert result["mean"].shape == (13,)
    assert result["std"].shape == (13,)
    assert np.all(np.isfinite(result["values"]))


def test_spectral_features_are_finite_and_non_negative() -> None:
    result = calculate_spectral_features(_sine(), SAMPLE_RATE)

    assert set(result) == {
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_hz",
        "zero_crossing_rate",
    }
    assert all(np.isfinite(value) and value >= 0 for value in result.values())

