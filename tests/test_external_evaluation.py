"""Локальные синтетические проверки внешнего теста; это не результаты CNN на RVC."""

import json
import shutil
import subprocess
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("tensorboard")

from audio_deepfake_detector.src import external_evaluation as external
from audio_deepfake_detector.src.cnn_training import AudioDeepfakeCNN, MelConfig


@pytest.fixture
def folders(tmp_path):
    return external.ensure_external_folders(tmp_path / "external")


@pytest.mark.parametrize("seconds, expected", [(0.3, 1), (4, 1), (10, 3), (18, 5)])
@pytest.mark.parametrize("rate", [16_000, 32_000, 44_100, 48_000])
def test_segment_lengths_rates_and_channels(folders, tmp_path, seconds, expected, rate):
    wave = np.zeros((round(seconds * rate), 2), dtype=np.float32)
    wave[:, 0] = .1
    sf.write(folders["REAL"] / "voice.wav", wave, rate)
    files, errors = external.scan_external_files(tmp_path / "external")
    assert errors.empty
    rows, failures = external.prepare_segments(files, tmp_path / "prepared", sample_rate=32_000)
    assert failures.empty
    assert len(rows) == expected
    assert rows["sample_id"].nunique() == 1
    assert all(sf.info(tmp_path / "prepared" / name).frames == 128_000 for name in rows["audio_path"])
    assert all(sf.info(tmp_path / "prepared" / name).samplerate == 32_000 for name in rows["audio_path"])
    assert rows.iloc[-1]["valid_seconds"] == pytest.approx(seconds % 4 or 4, abs=1 / rate)


def test_recursive_names_duplicates_and_bad_audio(folders, tmp_path):
    (folders["REAL"] / "speaker1").mkdir()
    (folders["REAL"] / "speaker2").mkdir()
    sf.write(folders["REAL"] / "speaker1" / "voice.wav", np.ones(4000) * .1, 16_000)
    sf.write(folders["REAL"] / "speaker2" / "voice.wav", np.ones(4000) * .2, 16_000)
    (folders["FAKE"] / "broken.wav").write_bytes(b"not audio")
    files, errors = external.scan_external_files(tmp_path / "external")
    assert len(files) == 2 and len(errors) == 1
    assert files["source_file_id"].nunique() == 2
    shutil.copyfile(folders["REAL"] / "speaker1" / "voice.wav", folders["FAKE"] / "copy.wav")
    with pytest.raises(ValueError, match="Одинаковое содержимое"):
        external.scan_external_files(tmp_path / "external")


def test_empty_and_metrics_na(folders, tmp_path):
    files, errors = external.scan_external_files(tmp_path / "external")
    assert files.empty and errors.empty
    table = pd.DataFrame({"true_label": [0], "fake_score": [.8]})
    metrics = external.safe_metrics(table, .5)
    assert metrics["accuracy"] == 0
    assert metrics["fp"] == 1
    assert metrics["eer"] is None and metrics["roc_auc"] is None
    assert metrics["f1"] is None


def test_aggregation_and_confusion():
    rows = pd.DataFrame([
        {"sample_id": "A", "segment_id": "A0", "label": "real", "relative_path": "REAL/a.wav", "source_path": "/a", "group": "speaker"},
        {"sample_id": "A", "segment_id": "A1", "label": "real", "relative_path": "REAL/a.wav", "source_path": "/a", "group": "speaker"},
        {"sample_id": "B", "segment_id": "B0", "label": "fake", "relative_path": "FAKE/b.wav", "source_path": "/b", "group": "rvc"},
    ])
    segments, files = external.prediction_tables(rows, [.1, .7, .8], .5)
    assert len(segments) == 3 and len(files) == 2
    assert files.iloc[0]["fake_score"] == pytest.approx(.4)
    stats = external.safe_metrics(files, .5)
    assert [stats[key] for key in ("tn", "fp", "fn", "tp")] == [1, 0, 0, 1]
    assert [stats[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc")] == [1] * 5
    assert stats["eer"] == 0


def test_external_inference_save_and_immutable_checkpoint(folders, tmp_path):
    for label, value in (("REAL", .1), ("FAKE", .2)):
        sf.write(folders[label] / f"{label}.flac", np.ones(4000) * value, 32_000)
    files, errors = external.scan_external_files(tmp_path / "external")
    assert errors.empty
    checkpoint = tmp_path / "best_model.pth"
    torch.save({"class_mapping": {"real": 0, "fake": 1}, "mel_config": asdict(MelConfig()),
                "training_config": {"threshold": .5}, "epoch": 7, "best_epoch": 7,
                "model_state": AudioDeepfakeCNN().state_dict()}, checkpoint)
    prep = tmp_path / "preprocessing_config.json"
    prep.write_text(json.dumps({"target_sample_rate": 32_000, "segment_seconds": 4,
                                "output_format": "FLAC", "flac_subtype": "PCM_16"}))
    before = external._sha256(checkpoint)
    result = external.evaluate_external(files, checkpoint, prep, batch_size=2, git_commit="synthetic")
    assert external._sha256(checkpoint) == before
    assert result["config"]["checkpoint_epoch"] == 8
    assert result["config"]["aggregation"].startswith("mean sigmoid")
    figures = external.external_figures(result)
    output = external.save_external(result, figures, folders["results"])
    assert (output / "file_predictions.csv").exists()
    assert (output / "confusion_matrix.png").exists()
    assert (output / "evaluation_summary.md").exists()
    assert external.save_external(result, figures, folders["results"]) != output
    prep.write_text(json.dumps({"target_sample_rate": 16_000, "segment_seconds": 4,
                                "output_format": "FLAC", "flac_subtype": "PCM_16"}))
    with pytest.raises(ValueError, match="не соответствует"):
        external.evaluate_external(files, checkpoint, prep)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
@pytest.mark.parametrize("extension", ["mp3", "m4a"])
def test_compressed_audio_when_decoder_available(folders, tmp_path, extension):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000), 16_000)
    target = folders["FAKE"] / f"test.{extension}"
    process = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), str(target)],
                             capture_output=True, check=False)
    if process.returncode:
        pytest.skip(f"ffmpeg не поддерживает {extension}")
    waveform, rate = external.decode_audio(target)
    assert rate == 16_000 and waveform.size > 0
