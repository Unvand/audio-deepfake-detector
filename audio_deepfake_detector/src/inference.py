"""Точка расширения для будущего подключения обученной CNN."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """Результат, который GUI сможет показать без изменения своей структуры."""

    verdict: str
    confidence: float


class ModelNotLoadedError(RuntimeError):
    """Модель ещё не обучена или не загружена."""


class AudioDeepfakeInference:
    """Интерфейс будущего CNN-инференса.

    Сейчас класс намеренно не загружает модель и не генерирует mock-вердикт.
    После обучения CNN здесь можно реализовать загрузку весов и метод ``predict``.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._model = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def predict(self, audio: np.ndarray, sample_rate: int) -> InferenceResult:
        """Вернуть REAL/FAKE после будущего подключения модели."""
        del audio, sample_rate
        raise ModelNotLoadedError("CNN-модель не загружена")

