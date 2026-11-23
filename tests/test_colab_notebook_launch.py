"""Проверка запуска длительных операций из ячеек notebook 03."""

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "audio_deepfake_detector/notebooks/03_ruasd_train_cnn.ipynb"
)


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _launch_cell(flag: str) -> str:
    cell, = (cell for cell in _code_cells() if f"{flag} = False" in cell.splitlines())
    return cell


def _execute(cell: str, namespace: dict) -> None:
    exec(compile(cell, str(NOTEBOOK), "exec"), namespace)  # noqa: S102


@pytest.mark.parametrize(
    ("flag", "operation"),
    [("RUN_TRAINING", "train_cnn"), ("RUN_CACHE_PREPARATION", "prepare_local_cache")],
)
def test_operation_runs_only_in_its_opt_in_cell(flag: str, operation: str) -> None:
    launch = _launch_cell(flag)
    calls = [
        cell
        for cell in _code_cells()
        for node in ast.walk(ast.parse(cell))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == operation
    ]
    assert calls == [launch]
    operation_mock = Mock()
    _execute(launch, {operation: operation_mock})
    operation_mock.assert_not_called()


def test_training_cell_uses_settings_at_launch() -> None:
    cell = _launch_cell("RUN_TRAINING").replace("RUN_TRAINING = False", "RUN_TRAINING = True")
    settings = SimpleNamespace(value=("old mel", "old training"))
    namespace = {
        "active_data_root": Mock(return_value=Path("/content/cache")),
        "current_configs": lambda: settings.value,
        "resolve_run_directory": lambda root, run_id: root / "runs" / run_id,
        "TRAINING_ROOT": Path("/content/drive/training"),
        "run_id_widget": SimpleNamespace(value="old_run"),
        "resume_widget": SimpleNamespace(value=False),
        "train_cnn": Mock(return_value=(object(), [1, 2])),
    }
    settings.value = ("new mel", "new training")
    namespace["run_id_widget"].value = "resume_run"
    namespace["resume_widget"].value = True
    _execute(cell, namespace)
    namespace["active_data_root"].assert_called_once_with(("train", "validation"))
    namespace["train_cnn"].assert_called_once_with(
        Path("/content/cache"),
        Path("/content/drive/training/runs/resume_run"),
        "new mel",
        "new training",
        resume=True,
    )
    assert namespace["history"] == [1, 2]


@dataclass
class _CacheResult:
    ready: bool


def test_cache_cell_uses_selected_splits_and_updates_status() -> None:
    cell = _launch_cell("RUN_CACHE_PREPARATION").replace(
        "RUN_CACHE_PREPARATION = False", "RUN_CACHE_PREPARATION = True"
    )
    selected = ("train", "validation")
    progress = Mock()
    prepare = Mock(return_value=_CacheResult(False))
    verified = {"train", "test_unseen"}
    namespace = {
        "selected_cache_splits": lambda: selected,
        "CACHE_VERIFIED_SPLITS": verified,
        "update_copy_progress": progress,
        "prepare_local_cache": prepare,
        "DRIVE_PREPARED_ROOT": Path("/content/drive/prepared"),
        "LOCAL_CACHE_ROOT": Path("/content/cache"),
        "checksum_widget": SimpleNamespace(value=True),
        "asdict": asdict,
        "display": Mock(),
        "pd": pd,
    }
    with pytest.raises(RuntimeError, match="Кэш не готов"):
        _execute(cell, namespace)
    assert verified == {"test_unseen"}
    prepare.assert_called_once_with(
        Path("/content/drive/prepared"),
        Path("/content/cache"),
        selected,
        verify_checksums=True,
        progress_callback=progress,
    )
    prepare.return_value = _CacheResult(True)
    _execute(cell, namespace)
    assert verified == {"train", "validation", "test_unseen"}
