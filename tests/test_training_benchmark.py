"""Проверка измеримого benchmark без утверждений о реальном RuASD."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("tensorboard")

from audio_deepfake_detector.src.cnn_training import MelConfig, TrainingConfig
from audio_deepfake_detector.src.training_benchmark import benchmark_training_pipeline


def test_benchmark_reports_all_pipeline_stages(tmp_path: Path) -> None:
    rows = []
    time_axis = np.arange(16_000, dtype=np.float32) / 32_000
    for index, label in enumerate(["real", "fake"] * 4):
        relative = Path("train") / f"segment-{index}.flac"
        (tmp_path / "train").mkdir(exist_ok=True)
        waveform = (0.1 * np.sin(2 * np.pi * (220 + index) * time_axis)).astype(
            np.float32
        )
        sf.write(tmp_path / relative, waveform, 32_000, format="FLAC")
        rows.append(
            {
                "segment_id": f"segment-{index}",
                "sample_id": f"sample-{index}",
                "label": label,
                "split": "train",
                "subset": "synthetic",
                "audio_path": relative.as_posix(),
            }
        )
    manifest = pd.DataFrame(rows)
    result = benchmark_training_pipeline(
        manifest,
        tmp_path,
        MelConfig(n_fft=512, win_length=512, hop_length=320, n_mels=32),
        TrainingConfig(batch_size=8, num_workers=0, use_amp=False),
        sample_count=4,
        batch_count=1,
    )
    expected = {
        "raw_file_read_ms_per_file",
        "flac_decode_ms_per_file",
        "log_mel_cpu_ms_per_file",
        "batch_load_ms",
        "host_to_device_ms",
        "cnn_forward_ms",
    }
    assert expected.issubset(result)
    assert all(result[key] >= 0 for key in expected)
    assert result["measured_batches"] == 1
    assert result["device"] in {"cpu", "cuda"}
