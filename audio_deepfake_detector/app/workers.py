"""Фоновая загрузка и анализ аудио без блокировки интерфейса."""

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile
from mutagen import MutagenError
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from audio_deepfake_detector.src.audio_analysis import (
    calculate_f0,
    calculate_mfcc,
    calculate_spectral_features,
)
from audio_deepfake_detector.src.inference import (
    AudioDeepfakeInference,
    ModelNotLoadedError,
)
from audio_deepfake_detector.src.preprocessing import TARGET_SAMPLE_RATE, load_audio

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def read_audio_metadata(file_path: str | Path) -> dict[str, Any]:
    """Получить доступную техническую информацию без падения из-за bitrate."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Поддерживаются WAV, FLAC, MP3 и OGG")

    metadata: dict[str, Any] = {
        "name": path.name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "duration": None,
        "sample_rate": None,
        "channels": None,
        "format": path.suffix.lstrip(".").upper(),
        "subtype": None,
        "bitrate": None,
    }

    soundfile_error: Exception | None = None
    try:
        info = sf.info(str(path))
        metadata.update(
            duration=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            format=info.format or metadata["format"],
            subtype=info.subtype,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        soundfile_error = exc

    try:
        media = MutagenFile(str(path))
        if media is not None and getattr(media, "info", None) is not None:
            info = media.info
            metadata["duration"] = metadata["duration"] or float(info.length)
            metadata["sample_rate"] = metadata["sample_rate"] or getattr(
                info, "sample_rate", None
            )
            metadata["channels"] = metadata["channels"] or getattr(
                info, "channels", None
            )
            metadata["bitrate"] = getattr(info, "bitrate", None)
    except (MutagenError, OSError, RuntimeError, ValueError) as exc:
        # Bitrate — необязательное поле, его отсутствие не мешает анализу.
        metadata["bitrate_error"] = str(exc)

    if metadata["duration"] is None and soundfile_error is not None:
        raise ValueError(f"Не удалось прочитать аудиофайл: {soundfile_error}")
    return metadata


class AnalysisWorker(QObject):
    """Выполняет описательный акустический анализ в отдельном QThread."""

    progress = pyqtSignal(int)
    stage_changed = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self._cancel_requested = Event()

    def cancel(self) -> None:
        """Потокобезопасно запросить остановку между этапами анализа."""
        self._cancel_requested.set()

    def _stop_if_cancelled(self) -> bool:
        """Сообщить об отмене и вернуть ``True``, если она запрошена."""
        if not self._cancel_requested.is_set():
            return False
        self.cancelled.emit()
        return True

    @pyqtSlot()
    def run(self) -> None:
        try:
            result: dict[str, Any] = {"errors": {}}

            self.stage_changed.emit("Чтение технической информации…")
            result["metadata"] = read_audio_metadata(self.file_path)
            self.progress.emit(10)
            if self._stop_if_cancelled():
                return

            self.stage_changed.emit("Загрузка и приведение аудио к mono 16 кГц…")
            audio, sample_rate = load_audio(self.file_path, TARGET_SAMPLE_RATE)
            if audio.size == 0:
                raise ValueError("Аудиофайл не содержит отсчётов")
            result["audio"] = audio
            result["sample_rate"] = sample_rate
            result["duration_processed"] = audio.size / sample_rate
            self.progress.emit(30)
            if self._stop_if_cancelled():
                return

            self.stage_changed.emit("Расчёт F0 / Pitch…")
            result["f0"] = self._safe_feature(
                "f0", calculate_f0, audio, sample_rate, result["errors"]
            )
            self.progress.emit(55)
            if self._stop_if_cancelled():
                return

            self.stage_changed.emit("Расчёт MFCC…")
            result["mfcc"] = self._safe_feature(
                "mfcc", calculate_mfcc, audio, sample_rate, result["errors"]
            )
            self.progress.emit(72)
            if self._stop_if_cancelled():
                return

            self.stage_changed.emit("Расчёт спектральных характеристик…")
            result["spectral"] = self._safe_feature(
                "spectral",
                calculate_spectral_features,
                audio,
                sample_rate,
                result["errors"],
            )
            self.progress.emit(90)
            if self._stop_if_cancelled():
                return

            self.stage_changed.emit("Проверка CNN-модели…")
            result["cnn"] = self._run_cnn_placeholder(audio, sample_rate)
            self.progress.emit(100)
            self.stage_changed.emit("Анализ завершён")
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — граница фонового Qt-worker
            self.failed.emit(str(exc))

    @staticmethod
    def _safe_feature(
        name: str,
        calculation: Callable[[np.ndarray, int], dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        errors: dict[str, str],
    ) -> dict[str, Any] | None:
        """Выполнить независимый расчёт, сохранив локальную ошибку."""
        try:
            return calculation(audio, sample_rate)
        except Exception as exc:  # noqa: BLE001 — независимая сторонняя функция
            errors[name] = str(exc)
            return None

    @staticmethod
    def _run_cnn_placeholder(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        inference = AudioDeepfakeInference()
        try:
            prediction = inference.predict(audio, sample_rate)
            return {
                "model_loaded": True,
                "verdict": prediction.verdict,
                "confidence": prediction.confidence,
            }
        except ModelNotLoadedError:
            return {
                "model_loaded": False,
                "verdict": "НЕ ВЫПОЛНЕН",
                "confidence": None,
            }
