"""Проверка оценки на синтетических FLAC без обучения модели."""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("tensorboard")

from audio_deepfake_detector.src import ruasd_evaluation as evaluation
from audio_deepfake_detector.src.cnn_training import AudioDeepfakeCNN, MelConfig


def test_metrics_and_sample_aggregation():
    rows = pd.DataFrame(
        {
            "segment_id": list("abcde"),
            "sample_id": ["A", "A", "B", "C", "D"],
            "subset": ["REAL1", "REAL1", "REAL2", "TTS1", "TTS2"],
            "label": ["real", "real", "real", "fake", "fake"],
            "audio_path": [f"test_unseen/{name}.flac" for name in "abcde"],
        }
    )
    segments, source = evaluation.prediction_tables(
        rows, [0.1, 0.9, 0.7, 0.8, 0.6], 0.65
    )
    assert len(segments) == 5
    assert source.loc[0, "fake_score"] == pytest.approx(0.5)
    assert source.loc[0, "n_segments"] == 2
    metrics = evaluation.evaluation_metrics(source, 0.65)
    assert [metrics[key] for key in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    for key in ("accuracy", "precision", "recall", "f1", "eer", "far_fpr", "frr_fnr"):
        assert metrics[key] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)


@pytest.fixture
def prepared(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    config = tmp_path / "preprocessing_config.json"
    root = tmp_path / "prepared"
    (root / "test_unseen").mkdir(parents=True)
    prep = {"target_sample_rate": 32000, "segment_seconds": 4, "output_format": "FLAC"}
    config.write_text(json.dumps(prep), encoding="utf-8")
    rows = []
    for idx, (sample, label, subset) in enumerate(
        (
            ("r", "real", "CommonVoice"),
            ("f", "fake", "ruxTTS"),
            ("f", "fake", "ruxTTS"),
        )
    ):
        name = f"test_unseen/{idx}.flac"
        sf.write(root / name, np.zeros(128000), 32000)
        rows.append(
            {
                "segment_id": str(idx),
                "sample_id": sample,
                "label": label,
                "subset": subset,
                "split": "test_unseen",
                "audio_path": name,
            }
        )
    # Отсутствующие FLAC другого split не должны читаться.
    rows.append(
        {
            "segment_id": "train0",
            "sample_id": "train0",
            "label": "fake",
            "subset": "train_model",
            "split": "train",
            "audio_path": "train/not_present.flac",
        }
    )
    pd.DataFrame(rows).to_csv(root / evaluation.MANIFEST_NAME, index=False)
    state = {
        key: torch.zeros_like(value)
        for key, value in AudioDeepfakeCNN().state_dict().items()
    }
    torch.save(
        {
            "class_mapping": {"real": 0, "fake": 1},
            "mel_config": asdict(MelConfig()),
            "training_config": {"threshold": 0.8},
            "epoch": 7,
            "best_epoch": 7,
            "model_state": state,
            "history": [{"epoch": 7, "validation_source": {"f1": 0.9}}],
        },
        checkpoint,
    )
    return root, checkpoint, config


def test_checkpoint_configuration_and_split_guard(prepared):
    root, checkpoint, _config = prepared
    with pytest.raises(ValueError, match="лучшая эпоха"):
        evaluation.load_best_model(checkpoint, expected_epoch=7)
    with pytest.raises(ValueError, match="best_model.pth"):
        evaluation.load_best_model(checkpoint.with_name("last_checkpoint.pth"))
    saved = torch.load(checkpoint, weights_only=False)
    correct_state = saved["model_state"]
    saved["model_state"] = {}
    torch.save(saved, checkpoint)
    with pytest.raises(RuntimeError, match="state_dict"):
        evaluation.load_best_model(checkpoint)
    saved["model_state"] = correct_state
    saved["mel_config"]["sample_rate"] = 16000
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="32 000"):
        evaluation.load_best_model(checkpoint)
    manifest = pd.read_csv(root / evaluation.MANIFEST_NAME)
    manifest.loc[manifest["split"].eq("train"), "sample_id"] = "r"
    manifest.to_csv(root / evaluation.MANIFEST_NAME, index=False)
    with pytest.raises(ValueError, match="общие sample_id"):
        evaluation.read_test_manifest(root)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_read_only_inference_and_saved_artifacts(
    prepared, tmp_path, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA отсутствует")
    root, checkpoint, config = prepared
    original = checkpoint.read_bytes()
    original_model_loader = evaluation.load_best_model
    observed = []

    def inspect_load(*args, **kwargs):
        model, *rest = original_model_loader(*args, **kwargs)

        def assert_inference(module, inputs):
            assert not module.training and torch.is_inference_mode_enabled()
            assert not any(param.requires_grad for param in module.parameters())
            observed.append(inputs[0].device.type)

        model.register_forward_pre_hook(assert_inference)
        return (model, *rest)

    monkeypatch.setattr(evaluation, "load_best_model", inspect_load)
    result = evaluation.evaluate_unseen(
        root,
        checkpoint,
        config,
        expected_epoch=8,
        device=device,
        git_commit="test-commit",
        run_id="baseline_cached_v1",
    )
    assert observed == [device]
    assert result["config"]["threshold"] == 0.8
    assert result["metrics"]["source"]["fn"] == 1
    assert result["metrics"]["segment"]["fn"] == 2
    by_subset = result["tables"]["metrics_by_subset"].set_index("subset")
    assert by_subset.loc["ruxTTS", "accuracy"] == 0
    assert by_subset.loc["CommonVoice", "accuracy"] == 1
    assert result["tables"]["false_negatives"].iloc[0]["n_segments"] == 2
    assert result["comparison"].loc["f1", "validation"] == 0.9
    figures = evaluation.evaluation_figures(result)
    output = evaluation.save_evaluation(
        result, figures, tmp_path / "evaluation" / "baseline"
    )
    before = (output / "test_metrics.json").read_bytes()
    assert evaluation.save_evaluation(result, {}, output).parent == output
    assert (output / "test_metrics.json").read_bytes() == before
    assert (output / "confusion_matrix.png").stat().st_size > 0
    assert (output / "roc_curve.png").stat().st_size > 0
    assert (output / "segment_predictions.csv").is_file()
    assert "ruxTTS" in (output / "evaluation_summary.md").read_text(encoding="utf-8")
    assert checkpoint.read_bytes() == original
    for fig in figures.values():
        plt.close(fig)


def test_notebook_is_opt_in():
    path = (
        Path(__file__).resolve().parents[1]
        / "audio_deepfake_detector/notebooks/06_ruasd_evaluate_cnn.ipynb"
    )
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    source = "".join(next(cell["source"] for cell in cells if cell["id"] == "evaluate"))
    namespace = {}
    exec(source, namespace)  # noqa: S102
    assert namespace["RUN_EVALUATION"] is False
    assert "result" not in namespace
