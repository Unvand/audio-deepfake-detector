"""Тесты обработки ошибок, коротких файлов и отсутствующей модели."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_deepfake_detector.app.workers import AnalysisWorker, read_audio_metadata
from audio_deepfake_detector.src.inference import (
    AudioDeepfakeInference,
    ModelNotLoadedError,
)


def test_too_short_file_does_not_crash_full_analysis(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(32, dtype=np.float32), 16_000, subtype="FLOAT")
    outcome: dict[str, object] = {}
    worker = AnalysisWorker(str(path))
    worker.finished.connect(lambda result: outcome.setdefault("result", result))
    worker.failed.connect(lambda message: outcome.setdefault("error", message))

    worker.run()

    assert "error" not in outcome
    result = outcome["result"]
    assert isinstance(result, dict)
    assert result["cnn"]["model_loaded"] is False
    assert result["cnn"]["verdict"] == "НЕ ВЫПОЛНЕН"


def test_corrupted_file_metadata_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="Не удалось прочитать аудиофайл"):
        read_audio_metadata(path)


def test_empty_models_directory_keeps_inference_unloaded(tmp_path: Path) -> None:
    models_directory = tmp_path / "models"
    models_directory.mkdir()
    inference = AudioDeepfakeInference(models_directory / "model.pt")

    assert inference.is_model_loaded is False
    with pytest.raises(ModelNotLoadedError, match="не загружена"):
        inference.predict(np.zeros(100, dtype=np.float32), 16_000)


def test_worker_never_generates_fake_real_or_fake_verdict() -> None:
    status = AnalysisWorker._run_cnn_placeholder(
        np.zeros(100, dtype=np.float32),
        16_000,
    )

    assert status["model_loaded"] is False
    assert status["verdict"] not in {"REAL", "FAKE"}
    assert status["confidence"] is None

