"""Фоновая загрузка и анализ аудио без блокировки интерфейса."""

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.inference import AudioDeepfakeInference, ModelNotLoadedError
from src.preprocessing import TARGET_SAMPLE_RATE, load_audio


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
    except Exception as exc:  # MP3 может зависеть от сборки libsndfile.
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
    except Exception:
        # Bitrate — необязательное поле, его отсутствие не мешает анализу.
        pass

    if metadata["duration"] is None and soundfile_error is not None:
        raise ValueError(f"Не удалось прочитать аудиофайл: {soundfile_error}")
    return metadata


class AnalysisWorker(QObject):
    """Выполняет описательный акустический анализ в отдельном QThread."""

    progress = pyqtSignal(int)
    stage_changed = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            result: dict[str, Any] = {"errors": {}}

            self.stage_changed.emit("Чтение технической информации…")
            result["metadata"] = read_audio_metadata(self.file_path)
            self.progress.emit(10)

            self.stage_changed.emit("Загрузка и приведение аудио к mono 16 кГц…")
            audio, sample_rate = load_audio(self.file_path, TARGET_SAMPLE_RATE)
            if audio.size == 0:
                raise ValueError("Аудиофайл не содержит отсчётов")
            result["audio"] = audio
            result["sample_rate"] = sample_rate
            result["duration_processed"] = audio.size / sample_rate
            self.progress.emit(30)

            self.stage_changed.emit("Расчёт F0 / Pitch…")
            result["f0"] = self._calculate_f0(audio, sample_rate, result["errors"])
            self.progress.emit(55)

            self.stage_changed.emit("Расчёт MFCC…")
            result["mfcc"] = self._calculate_mfcc(audio, sample_rate, result["errors"])
            self.progress.emit(72)

            self.stage_changed.emit("Расчёт спектральных характеристик…")
            result["spectral"] = self._calculate_spectral(
                audio, sample_rate, result["errors"]
            )
            self.progress.emit(90)

            self.stage_changed.emit("Проверка CNN-модели…")
            result["cnn"] = self._run_cnn_placeholder(audio, sample_rate)
            self.progress.emit(100)
            self.stage_changed.emit("Анализ завершён")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _calculate_f0(
        audio: np.ndarray, sample_rate: int, errors: dict[str, str]
    ) -> dict[str, Any] | None:
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sample_rate,
            )
            times = librosa.times_like(f0, sr=sample_rate)
            voiced_f0 = f0[np.isfinite(f0)]
            if voiced_f0.size == 0:
                raise ValueError("не удалось обнаружить voiced-фрагменты")
            return {
                "times": times,
                "values": f0,
                "mean": float(np.mean(voiced_f0)),
                "min": float(np.min(voiced_f0)),
                "max": float(np.max(voiced_f0)),
                "std": float(np.std(voiced_f0)),
                "voiced_ratio": float(np.mean(voiced_flag)),
            }
        except Exception as exc:
            errors["f0"] = str(exc)
            return None

    @staticmethod
    def _calculate_mfcc(
        audio: np.ndarray, sample_rate: int, errors: dict[str, str]
    ) -> dict[str, Any] | None:
        try:
            # 13 коэффициентов используются только для описательной визуализации.
            values = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            return {
                "values": values,
                "mean": np.mean(values, axis=1),
                "std": np.std(values, axis=1),
            }
        except Exception as exc:
            errors["mfcc"] = str(exc)
            return None

    @staticmethod
    def _calculate_spectral(
        audio: np.ndarray, sample_rate: int, errors: dict[str, str]
    ) -> dict[str, float] | None:
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            zcr = librosa.feature.zero_crossing_rate(audio)
            return {
                "spectral_centroid_hz": float(np.mean(centroid)),
                "spectral_bandwidth_hz": float(np.mean(bandwidth)),
                "spectral_rolloff_hz": float(np.mean(rolloff)),
                "zero_crossing_rate": float(np.mean(zcr)),
            }
        except Exception as exc:
            errors["spectral"] = str(exc)
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
