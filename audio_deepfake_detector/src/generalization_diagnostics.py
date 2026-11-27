"""Диагностика доменного сдвига без изменения обученной CNN."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft

from audio_deepfake_detector.src.external_evaluation import decode_audio
from audio_deepfake_detector.src.ruasd_preparation import segment_with_padding


def audio_features(waveform: np.ndarray, rate: int, *, silence_db: float = -40) -> dict:
    """Вычислить сопоставимые технические признаки mono-сигнала."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1 or not x.size or rate <= 0 or not np.isfinite(x).all():
        raise ValueError("Нужен непустой конечный mono-сигнал и корректная частота")
    magnitude = np.abs(stft(x, fs=rate, nperseg=min(1024, len(x)))[2])
    frequencies = np.linspace(0, rate / 2, magnitude.shape[0])
    power = magnitude.sum(axis=0)
    centroid = (frequencies[:, None] * magnitude).sum(axis=0) / np.maximum(power, 1e-12)
    cumulative = np.cumsum(magnitude, axis=0)
    rolloff = frequencies[(cumulative >= .85 * power).argmax(axis=0)]
    flatness = np.exp(np.log(magnitude + 1e-12).mean(axis=0)) / np.maximum(magnitude.mean(axis=0), 1e-12)
    threshold = 10 ** (silence_db / 20)
    silent = np.abs(x) < threshold
    # Длины пауз получаем по последовательным переходам, а не по числу тихих отсчётов.
    edges = np.flatnonzero(np.diff(np.r_[0, silent.astype(int), 0]))
    pauses = edges[1::2] - edges[::2]
    return {
        "sample_rate": rate, "duration_seconds": len(x) / rate,
        "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(x))),
        "silence_fraction": float(silent.mean()), "near_zero_fraction": float((np.abs(x) < 1e-5).mean()),
        "longest_pause_seconds": float(pauses.max() / rate) if len(pauses) else 0.,
        "centroid_hz": float(np.mean(centroid)), "rolloff_hz": float(np.mean(rolloff)),
        "flatness": float(np.mean(flatness)),
        "zcr": float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if len(x) > 1 else 0.,
        "start_rms": float(np.sqrt(np.mean(x[:min(rate, len(x))] ** 2))),
        "end_rms": float(np.sqrt(np.mean(x[-min(rate, len(x)):] ** 2))),
    }


def padding_features(waveform: np.ndarray, rate: int, seconds: float = 4.) -> dict:
    """Оценить долю нулевого дополнения последнего сегмента."""
    parts = segment_with_padding(waveform, rate, seconds)
    total = len(parts) * round(rate * seconds)
    return {"segments": len(parts), "last_incomplete": int(parts[-1][1] < seconds),
            "padding_fraction": (total - len(waveform)) / total}


def transform_audio(waveform: np.ndarray, rate: int, variant: str, *, seed: int = 42) -> np.ndarray:
    """Преобразовать временную копию; входной массив не изменяется."""
    x = np.asarray(waveform, dtype=np.float32).copy()
    if variant == "baseline" or variant == "format":
        return x
    if variant == "volume":
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        return np.clip(x * (.05 / max(rms, 1e-8)), -1, 1).astype(np.float32)
    if variant == "noise":
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        noise = np.random.default_rng(seed).normal(0, max(rms, 1e-8) * 10 ** (-30 / 20), len(x))
        return np.clip(x + noise, -1, 1).astype(np.float32)
    if variant == "trim_pauses":
        frame = max(1, round(.02 * rate))
        chunks = [x[i:i + frame] for i in range(0, len(x), frame)]
        keep = [chunk for chunk in chunks if np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) >= .01]
        return np.concatenate(keep) if keep else x
    raise ValueError(f"Неизвестное преобразование: {variant}")


def inspect_file(path: Path, domain: str, label: str, identifier: str, *, valid_seconds: float | None = None) -> dict:
    """Прочитать один файл и вернуть только обезличенные признаки."""
    info = sf.info(path) if path.suffix.lower() == ".flac" else None
    waveform, rate = decode_audio(path)
    result = {"id": identifier, "domain": domain, "label": label,
              "channels": info.channels if info else None,
              "format": info.format if info else path.suffix.lower(),
              **audio_features(waveform, rate), **padding_features(waveform, rate)}
    if valid_seconds is not None:
        result["manifest_padding_fraction"] = max(0., 1 - valid_seconds / 4.)
    return result


def anonymous_id(value: str) -> str:
    """Стабильный идентификатор без имени и пути."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def save_diagnostics(output: Path, features: pd.DataFrame, scores: pd.DataFrame,
                     changes: pd.DataFrame, config: dict, report: str) -> None:
    """Сохранить обезличенные таблицы и конфигурацию в новом каталоге."""
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("audio_features", features), ("segment_scores", scores),
                        ("transformation_results", changes)):
        if any("path" in column.lower() or "speaker" in column.lower() for column in table.columns):
            raise ValueError("Приватные поля не допускаются в сохраняемой таблице")
        table.to_csv(output / f"{name}.csv", index=False)
    if any("input_sha256" in key or "path" in key for key in config):
        raise ValueError("Конфигурация содержит путь или частные хеши файлов")
    (output / "domain_comparison.csv").write_text(
        features.groupby(["domain", "label"]).agg(n=("id", "size"), median_rms=("rms", "median"),
        median_silence=("silence_fraction", "median"), median_flatness=("flatness", "median"))
        .to_csv(), encoding="utf-8")
    (output / "diagnostic_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (output / "diagnostic_report.md").write_text(report, encoding="utf-8")
