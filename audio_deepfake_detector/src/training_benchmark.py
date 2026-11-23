"""Диагностика узких мест pipeline без обучения и изменения модели."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from audio_deepfake_detector.src.cnn_training import (
    AudioDeepfakeCNN,
    MelConfig,
    RuASDSegmentDataset,
    TrainingConfig,
)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_training_pipeline(
    manifest: pd.DataFrame,
    prepared_root: str | Path,
    mel_config: MelConfig,
    training_config: TrainingConfig,
    *,
    sample_count: int = 16,
    batch_count: int = 5,
) -> dict[str, Any]:
    """Измерить I/O, decode, CPU Mel, DataLoader, GPU transfer и CNN forward.

    Функция не выполняет backward/optimizer step и не является обучением.
    Результаты относятся только к текущей среде и выбранному корню данных.
    """
    mel_config.validate()
    training_config.validate()
    root = Path(prepared_root)
    rows = manifest[manifest["split"] == "train"].head(sample_count)
    if rows.empty:
        raise ValueError("Нет train-записей для benchmark")
    raw_payloads: list[bytes] = []
    read_started = time.perf_counter()
    for relative in rows["audio_path"]:
        raw_payloads.append((root / str(relative)).read_bytes())
    read_seconds = time.perf_counter() - read_started

    decoded_waveforms: list[np.ndarray] = []
    decode_started = time.perf_counter()
    for payload in raw_payloads:
        waveform, sample_rate = sf.read(
            io.BytesIO(payload), dtype="float32", always_2d=False
        )
        if sample_rate != mel_config.sample_rate:
            raise ValueError(f"Benchmark обнаружил частоту {sample_rate} Гц")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        decoded_waveforms.append(np.asarray(waveform, dtype=np.float32))
    decode_seconds = time.perf_counter() - decode_started

    mel_transform = MelSpectrogram(
        sample_rate=mel_config.sample_rate,
        n_fft=mel_config.n_fft,
        win_length=mel_config.win_length,
        hop_length=mel_config.hop_length,
        f_min=mel_config.f_min,
        f_max=mel_config.f_max,
        n_mels=mel_config.n_mels,
        power=mel_config.power,
    )
    to_db = AmplitudeToDB(stype="power", top_db=80.0)
    mel_started = time.perf_counter()
    for waveform in decoded_waveforms:
        log_mel = to_db(mel_transform(torch.from_numpy(waveform)))
        _ = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-6)
    mel_seconds = time.perf_counter() - mel_started

    train_dataset = RuASDSegmentDataset(manifest, root, "train", mel_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=training_config.num_workers > 0,
    )
    loader_iterator = iter(train_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeCNN().to(device).eval()
    amp_enabled = bool(training_config.use_amp and device.type == "cuda")
    load_times: list[float] = []
    transfer_times: list[float] = []
    forward_times: list[float] = []
    measured_batches = 0
    for _ in range(min(batch_count, len(train_loader))):
        load_started = time.perf_counter()
        batch, _, _ = next(loader_iterator)
        load_times.append(time.perf_counter() - load_started)

        _synchronise(device)
        transfer_started = time.perf_counter()
        device_batch = batch.to(device, non_blocking=True)
        _synchronise(device)
        transfer_times.append(time.perf_counter() - transfer_started)

        _synchronise(device)
        forward_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            _ = model(device_batch)
        _synchronise(device)
        forward_times.append(time.perf_counter() - forward_started)
        measured_batches += 1

    files = len(rows)
    return {
        "data_root": str(root),
        "device": str(device),
        "sample_count": files,
        "batch_size": training_config.batch_size,
        "measured_batches": measured_batches,
        "raw_file_read_ms_per_file": 1000 * read_seconds / files,
        "flac_decode_ms_per_file": 1000 * decode_seconds / files,
        "log_mel_cpu_ms_per_file": 1000 * mel_seconds / files,
        "batch_load_ms": 1000 * float(np.mean(load_times)),
        "host_to_device_ms": 1000 * float(np.mean(transfer_times)),
        "cnn_forward_ms": 1000 * float(np.mean(forward_times)),
        "note": "Диагностика без backward; не является оценкой времени эпохи.",
    }


def benchmark_roots(
    manifest: pd.DataFrame,
    roots: dict[str, str | Path],
    mel_config: MelConfig,
    training_config: TrainingConfig,
    *,
    sample_count: int = 16,
    batch_count: int = 5,
) -> pd.DataFrame:
    """Сравнить Drive и локальный кэш одинаковым benchmark."""
    records = []
    for name, root in roots.items():
        record = benchmark_training_pipeline(
            manifest,
            root,
            mel_config,
            training_config,
            sample_count=sample_count,
            batch_count=batch_count,
        )
        record["source"] = name
        records.append(record)
    return pd.DataFrame(records).set_index("source")
