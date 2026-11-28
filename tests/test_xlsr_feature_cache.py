"""Синтетический frozen encoder: проверка кэша, resume и полного checkpoint."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch import nn

from audio_deepfake_detector.src.xlsr_feature_cache import (
    build_cache,
    encoder_digest,
    load_cache,
    train_cached_head,
    verify_cached_logits,
)
from audio_deepfake_detector.src.xlsr_training import XLSRConfig, XLSRDetector


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def forward(self, waveform, attention_mask):
        return SimpleNamespace(last_hidden_state=self.linear(waveform[:, ::320, None]))

    def _get_feature_vector_attention_mask(self, length, mask):
        return mask[:, ::320][:, :length].bool()


def test_cache_resume_integrity_and_checkpoint(tmp_path):
    rows = []
    for split in ("train", "validation"):
        (tmp_path / split).mkdir()
        for label in ("real", "fake"):
            name = f"{split}/{label}.flac"
            sf.write(tmp_path / name, np.ones(32000) * (.1 if label == "real" else -.1), 32000)
            rows.append({"segment_id": name, "sample_id": name, "split": split,
                         "label": label, "audio_path": name, "valid_duration_sec": 1.})
    frame = pd.DataFrame(rows)
    cfg = XLSRConfig(batch_size=1, epochs=1)
    model = XLSRDetector(SmallEncoder(), 4, cfg)
    cache_dir = tmp_path / "cache"
    first = build_cache(frame, tmp_path, cache_dir, model, cfg, shard_size=1, local_work=tmp_path / "work")
    second = build_cache(frame, tmp_path, cache_dir, model, cfg, shard_size=1)
    assert first == second
    cached = load_cache(cache_dir, frame, cfg, encoder_digest(model.encoder), shard_size=1)
    assert len(cached["train"][1]) == 2
    assert verify_cached_logits(model, frame, tmp_path, cached, cfg) < 1e-4
    shard = cache_dir / "train/part-0000.npz"
    shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="повреждён"):
        load_cache(cache_dir, frame, cfg, encoder_digest(model.encoder), shard_size=1)
    build_cache(frame, tmp_path, cache_dir, model, cfg, shard_size=1)
    cached = load_cache(cache_dir, frame, cfg, encoder_digest(model.encoder), shard_size=1)
    history = train_cached_head(model, cached, tmp_path / "run", cfg, cache_identity=first["identity"])
    assert len(history) == 1
    best = torch.load(tmp_path / "run/best_model.pth", weights_only=False)
    restored = XLSRDetector(SmallEncoder(), 4, cfg)
    restored.load_state_dict(best["model"], strict=True)
    assert best["epoch"] == 0
    with pytest.raises(FileExistsError):
        train_cached_head(model, cached, tmp_path / "run", cfg, cache_identity=first["identity"])
