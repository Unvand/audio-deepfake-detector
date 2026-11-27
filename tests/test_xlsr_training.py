"""Миниатюрные тесты XLS-R без загрузки многогигабайтных весов."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch import nn

from audio_deepfake_detector.src.xlsr_training import (
    WaveformDataset,
    XLSRConfig,
    XLSRDetector,
    audit_manifest,
    masked_mean,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleNamespace(layers=nn.ModuleList([nn.Linear(1, 4) for _ in range(3)]))
        self.layers = self.encoder.layers

    def forward(self, waveform, attention_mask):
        return SimpleNamespace(last_hidden_state=self.layers[0](waveform[..., None]))


def test_mask_pooling_and_freeze():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]]])
    assert torch.allclose(masked_mean(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2., 3.]]))
    model = XLSRDetector(TinyEncoder(), 4, XLSRConfig())
    assert not any(p.requires_grad for p in model.encoder.parameters())
    x = torch.randn(1, 8)
    loss = nn.BCEWithLogitsLoss()(model(x, torch.ones_like(x, dtype=torch.bool)), torch.ones(1))
    loss.backward()
    assert model.head[1].weight.grad is not None
    assert all(p.grad is None for p in model.encoder.parameters())
    model.set_mode("partial")
    assert any(p.requires_grad for p in model.encoder.layers[-1].parameters())
    assert not any(p.requires_grad for p in model.encoder.layers[0].parameters())


def test_manifest_and_waveform(tmp_path):
    rows = []
    for split, label in (("train", "real"), ("validation", "fake"), ("test_unseen", "real")):
        (tmp_path / split).mkdir()
        sf.write(tmp_path / split / "one.flac", np.zeros(32000), 32000)
        rows.append({"sample_id": split, "segment_id": split, "audio_path": f"{split}/one.flac",
                     "split": split, "label": label, "valid_duration_sec": 1.0})
    path = tmp_path / "ruasd_segments_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame, summary = audit_manifest(path)
    assert len(summary) == 3
    wave, mask, label, source = WaveformDataset(frame, tmp_path, "train", XLSRConfig())[0]
    assert len(wave) == 64000 and mask.sum() == 16000
    assert label == 0 and source == "train"
    rows[1]["sample_id"] = "train"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Утечка"):
        audit_manifest(path)
