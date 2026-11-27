"""Проверки диагностики на небольших искусственных сигналах."""

import hashlib

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from audio_deepfake_detector.src.generalization_diagnostics import (
    audio_features,
    inspect_file,
    padding_features,
    save_diagnostics,
    transform_audio,
)


def test_features_padding_and_unchanged_input(tmp_path):
    x = np.r_[np.zeros(16000), np.ones(16000) * .2].astype("float32")
    path = tmp_path / "voice.wav"
    sf.write(path, x, 16000)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    features = audio_features(x, 16000)
    assert features["rms"] == pytest.approx(np.sqrt(.02))
    assert features["peak"] == pytest.approx(.2)
    assert features["silence_fraction"] == pytest.approx(.5)
    assert features["longest_pause_seconds"] == pytest.approx(1)
    assert padding_features(x, 16000)["padding_fraction"] == .5
    for variant in ("baseline", "format", "volume", "noise", "trim_pauses"):
        assert transform_audio(x, 16000, variant).size > 0
    assert np.array_equal(x, np.r_[np.zeros(16000), np.ones(16000) * .2].astype("float32"))
    assert inspect_file(path, "External", "REAL", "anonymous")["duration_seconds"] == 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_private_columns_rejected(tmp_path):
    data = pd.DataFrame({"id": ["A"], "domain": ["External"], "label": ["REAL"],
                         "rms": [.1], "silence_fraction": [0.], "flatness": [.2]})
    with pytest.raises(ValueError, match="Приватные"):
        save_diagnostics(tmp_path / "out", data.assign(speaker=["name"]), data, data, {}, "report")
