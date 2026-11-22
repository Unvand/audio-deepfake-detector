"""Метрики бинарного аудиодетектора, не зависящие от PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    """Рассчитать EER интерполяцией в точке FPR ≈ FNR."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size != 2:
        return float("nan")
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    return float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1 и EER для класса FAKE=1."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "eer": equal_error_rate(labels, scores),
    }


def aggregate_source_scores(
    sample_ids: list[str], labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Усреднить вероятности сегментов для метрик исходных аудиозаписей."""
    frame = pd.DataFrame({"sample_id": sample_ids, "label": labels, "score": scores})
    inconsistent = frame.groupby("sample_id")["label"].nunique().gt(1)
    if inconsistent.any():
        raise ValueError("Один sample_id имеет разные метки")
    grouped = frame.groupby("sample_id", sort=False).agg(
        label=("label", "first"), score=("score", "mean")
    )
    return grouped["label"].to_numpy(), grouped["score"].to_numpy()
