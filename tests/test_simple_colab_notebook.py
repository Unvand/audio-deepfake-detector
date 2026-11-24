"""Проверка точек входа notebook 05 без Colab и без обучения."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def cells():
    notebook = (
        Path(__file__).resolve().parents[1]
        / "audio_deepfake_detector/notebooks/05_ruasd_train_simple.ipynb"
    )
    data = json.loads(notebook.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell["source"])
        for cell in data["cells"] if cell["cell_type"] == "code"
    }


def _execute(source, namespace):
    exec(compile(source, "notebook 05", "exec"), namespace)  # noqa: S102


def test_restore_is_opt_in_and_skips_verified_cache(cells):
    for name, source in cells.items():
        compile(source, name, "exec")
    restore = Mock(return_value={"restored_files": 4, "skipped_files": 0})
    status = SimpleNamespace(ready=True, valid_files=4)
    namespace = {
        "restore_tar_cache": restore, "cache_status": lambda: status,
        "source_root": Path("prepared"), "archive_root": Path("tar"), "cache_root": Path("cache"),
        "splits": ("train", "validation"), "require_ready_cache": Mock(),
    }
    _execute(cells["restore"], namespace)
    launch = cells["restore"].replace("RUN_RESTORE_CACHE = False", "RUN_RESTORE_CACHE = True")
    _execute(launch, namespace)
    restore.assert_not_called()
    status.ready = False
    _execute(launch, namespace)
    assert restore.call_args.args == (
        Path("prepared"), Path("tar"), Path("cache"), ("train", "validation"),
    )
    namespace["require_ready_cache"].assert_called_once()


def test_widget_configuration_and_opt_in_training(cells, tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    pytest.importorskip("tensorboard")
    pytest.importorskip("ipywidgets")
    trainer = Mock(return_value=(None, []))
    cache = tmp_path / "cache"
    namespace = {
        "training_root": tmp_path, "source_root": tmp_path / "source", "cache_root": cache,
        "splits": ("train", "validation"), "require_ready_cache": Mock(),
    }
    _execute(cells["settings"], namespace)
    namespace["train_cnn"] = trainer
    _execute(cells["train"], namespace)
    trainer.assert_not_called()
    run_dir = tmp_path / "runs" / "baseline_cached_v1"
    assert namespace["current_run"]()[0] == run_dir
    namespace["controls"]["epochs"].value = 23
    namespace["controls"]["batch_size"].value = 16
    namespace["mel_controls"]["n_mels"].value = 64
    launch = cells["train"].replace("RUN_TRAINING = False", "RUN_TRAINING = True")
    with pytest.raises(FileNotFoundError):
        _execute(launch, namespace)
    trainer.assert_not_called()
    namespace["inspect_training_run"] = Mock(return_value=3)
    _execute(launch, namespace)
    root, output, mel, training = trainer.call_args.args
    assert (root, output) == (cache, run_dir)
    assert training.epochs == 23 and training.batch_size == 16 and mel.n_mels == 64
    assert trainer.call_args.kwargs == {"resume": True}
    assert "Следующая эпоха 3/23" in capsys.readouterr().out
