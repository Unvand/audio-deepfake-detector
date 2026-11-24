"""Интеграционные тесты CNN pipeline на маленьких синтетических FLAC."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from audio_deepfake_detector.src import cnn_training
from audio_deepfake_detector.src.cnn_training import (
    MelConfig,
    RuASDSegmentDataset,
    TrainingConfig,
    assert_checkpoint_compatible,
    build_loss_functions,
    compute_pos_weight,
    configs_from_mapping,
    inspect_training_run,
    make_loaders,
    resolve_run_directory,
    train_cnn,
)


def _prepared_dataset(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    layout = [
        ("train", "real", 3),
        ("train", "fake", 3),
        ("validation", "real", 1),
        ("validation", "fake", 1),
        ("test_unseen", "real", 1),
        ("test_unseen", "fake", 1),
    ]
    sample_index = 0
    time_axis = np.arange(32_000, dtype=np.float32) / 32_000
    for split, label, count in layout:
        for _ in range(count):
            segment_id = f"segment-{sample_index}"
            relative = Path(split) / f"{segment_id}.flac"
            (root / split).mkdir(parents=True, exist_ok=True)
            frequency = 220 + sample_index * 20
            waveform = (0.1 * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)
            sf.write(root / relative, waveform, 32_000, format="FLAC", subtype="PCM_16")
            rows.append(
                {
                    "segment_id": segment_id,
                    "sample_id": f"sample-{sample_index}",
                    "label": label,
                    "split": split,
                    "subset": f"synthetic-{label}",
                    "audio_path": relative.as_posix(),
                }
            )
            sample_index += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "ruasd_segments_manifest.csv", index=False)
    return manifest


def _widget_values() -> dict[str, object]:
    return {
        "batch_size": 16,
        "epochs": 7,
        "learning_rate": 3e-4,
        "weight_decay": 2e-4,
        "early_stopping_patience": 3,
        "num_workers": 0,
        "seed": 123,
        "balance_classes": False,
        "use_amp": False,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "f_min": 30.0,
        "f_max": 15_000.0,
    }


def test_widget_values_create_real_configs() -> None:
    mel, training = configs_from_mapping(_widget_values())
    assert mel.n_fft == 512
    assert mel.n_mels == 64
    assert training.batch_size == 16
    assert training.learning_rate == pytest.approx(3e-4)
    assert training.balance_classes is False


def test_invalid_mel_configuration_is_rejected() -> None:
    values = _widget_values()
    values["win_length"] = 1_024
    values["n_fft"] = 512
    with pytest.raises(ValueError, match="win_length"):
        configs_from_mapping(values)
    values = _widget_values()
    values["f_max"] = 16_001.0
    with pytest.raises(ValueError, match="Найквиста"):
        configs_from_mapping(values)


def test_dataset_mel_loaders_and_test_unseen_isolation(tmp_path: Path) -> None:
    manifest = _prepared_dataset(tmp_path)
    mel = MelConfig(n_fft=512, win_length=512, hop_length=160, n_mels=64)
    training = TrainingConfig(batch_size=8, num_workers=0, use_amp=False)
    dataset = RuASDSegmentDataset(manifest, tmp_path, "train", mel)
    log_mel, label, sample_id = dataset[0]
    assert log_mel.shape[0:2] == (1, 64)
    assert torch.isfinite(log_mel).all()
    assert label.item() in {0.0, 1.0}
    assert sample_id.startswith("sample-")

    train_loader, validation_loader = make_loaders(manifest, tmp_path, mel, training)
    assert train_loader.dataset.split == "train"
    assert validation_loader.dataset.split == "validation"
    assert "test_unseen" not in {
        train_loader.dataset.split,
        validation_loader.dataset.split,
    }


def test_pos_weight_and_losses_use_only_train() -> None:
    manifest = pd.DataFrame(
        {
            "split": ["train"] * 6 + ["validation"] * 10 + ["test_unseen"] * 10,
            "label": ["real"] * 2 + ["fake"] * 4 + ["real"] * 20,
        }
    )
    assert compute_pos_weight(manifest) == pytest.approx(0.5)
    config = TrainingConfig(batch_size=8, num_workers=0, balance_classes=True)
    train_loss, validation_loss, applied, computed = build_loss_functions(
        manifest, config, torch.device("cpu")
    )
    assert applied == pytest.approx(0.5)
    assert computed == pytest.approx(0.5)
    assert train_loss.pos_weight.item() == pytest.approx(0.5)
    assert validation_loss.pos_weight is None


def test_checkpoint_compatibility_and_run_directory(tmp_path: Path) -> None:
    mel = MelConfig()
    training = TrainingConfig()
    checkpoint = {
        "mel_config": asdict(mel),
        "training_config": asdict(training),
    }
    assert_checkpoint_compatible(checkpoint, mel, training)
    assert_checkpoint_compatible(checkpoint, mel, replace(training, epochs=25))
    with pytest.raises(ValueError, match="новый run_id"):
        assert_checkpoint_compatible(
            checkpoint, mel, replace(training, learning_rate=3e-4)
        )
    assert resolve_run_directory(tmp_path, "baseline_v1") == tmp_path / "runs/baseline_v1"
    with pytest.raises(ValueError, match="run_id"):
        resolve_run_directory(tmp_path, "../escape")


def test_resume_preflight_preserves_existing_run(tmp_path: Path) -> None:
    mel = MelConfig()
    config = TrainingConfig()
    path = tmp_path / "last_checkpoint.pth"
    with pytest.raises(FileNotFoundError, match="выключите Resume"):
        inspect_training_run(tmp_path, mel, config, resume=True)
    assert inspect_training_run(tmp_path, mel, config, resume=False) == 1
    torch.save({
        "epoch": 1,
        "training_config": asdict(replace(config, epochs=2)),
        "mel_config": asdict(mel),
    }, path)
    previous = path.read_bytes()
    assert inspect_training_run(tmp_path, mel, config, resume=True) == 3
    with pytest.raises(ValueError, match="несовместим"):
        inspect_training_run(tmp_path, mel, replace(config, batch_size=16), resume=True)
    with pytest.raises(FileExistsError):
        inspect_training_run(tmp_path, mel, config, resume=False)
    assert path.read_bytes() == previous


def test_training_checkpoint_resume_tensorboard_and_real_images(tmp_path: Path, monkeypatch) -> None:
    prepared = tmp_path / "content" / "ruasd_prepared_cache"
    manifest = _prepared_dataset(prepared)
    run_dir = tmp_path / "google_drive" / "training_v1" / "runs" / "tiny"
    mel = MelConfig(n_fft=512, win_length=512, hop_length=320, n_mels=32)
    first_config = TrainingConfig(
        batch_size=8,
        epochs=1,
        num_workers=0,
        early_stopping_patience=3,
        balance_classes=True,
        use_amp=False,
        spectrogram_examples_per_class=3,
    )
    _, first_history = train_cnn(prepared, run_dir, mel, first_config, resume=True)
    assert len(first_history) == 1
    checkpoint = torch.load(run_dir / "last_checkpoint.pth", weights_only=False)
    assert checkpoint["epoch"] == 0
    assert checkpoint["pos_weight"] == pytest.approx(1.0)
    assert checkpoint["scaler_state"] == {}
    assert "rng_state" in checkpoint
    assert (run_dir / "best_model.pth").exists()
    assert (run_dir / "training_history.csv").exists()
    assert (run_dir / "experiment_config.json").exists()
    assert (run_dir / "best_epoch.json").exists()
    assert not (prepared / "last_checkpoint.pth").exists()

    _, resumed_history = train_cnn(
        prepared, run_dir, mel, replace(first_config, epochs=2), resume=True
    )
    assert len(resumed_history) == 2
    resumed = torch.load(run_dir / "last_checkpoint.pth", weights_only=False)
    assert resumed["epoch"] == 1

    event_files = list((run_dir / "tensorboard").rglob("events.out.tfevents.*"))
    assert event_files
    accumulator = EventAccumulator(str(run_dir / "tensorboard"))
    accumulator.Reload()
    scalar_tags = accumulator.Tags()["scalars"]
    image_tags = accumulator.Tags()["images"]
    assert "LearningRate/value" in scalar_tags
    assert "Timing/epoch_seconds" in scalar_tags
    assert "Experiment/pos_weight" in scalar_tags
    assert "ConfusionMatrix/validation_source" in image_tags
    assert len([tag for tag in image_tags if tag.startswith("Spectrograms/REAL")]) == 3
    assert len([tag for tag in image_tags if tag.startswith("Spectrograms/FAKE")]) == 3
    assert set(manifest[manifest["split"] == "test_unseen"]["sample_id"]).isdisjoint(
        set(manifest[manifest["split"] != "test_unseen"]["sample_id"])
    )

    class StopBeforeTraining(Exception):
        pass

    def check_next_epoch(*_args, **kwargs):
        assert kwargs["epoch"] == 2
        assert kwargs["total_epochs"] == 20
        raise StopBeforeTraining

    saved_bytes = (run_dir / "last_checkpoint.pth").read_bytes()
    monkeypatch.setattr(cnn_training, "_run_epoch", check_next_epoch)
    with pytest.raises(StopBeforeTraining):
        train_cnn(prepared, run_dir, mel, replace(first_config, epochs=20), resume=True)
    assert (run_dir / "last_checkpoint.pth").read_bytes() == saved_bytes
