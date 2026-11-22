"""Небольшие unit-тесты метрик CNN без запуска обучения."""

import numpy as np

from audio_deepfake_detector.src.training_metrics import (
    aggregate_source_scores,
    binary_metrics,
    equal_error_rate,
)


def test_binary_metrics_for_perfect_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(labels, scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert equal_error_rate(labels, scores) == 0.0


def test_source_scores_average_segments() -> None:
    labels, scores = aggregate_source_scores(
        ["a", "a", "b", "b"],
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.7, 0.9]),
    )
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_allclose(scores, [0.2, 0.8])
