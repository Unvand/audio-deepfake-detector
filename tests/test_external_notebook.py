"""Проверка структуры запуска notebook 07 без Colab и реальных записей."""

import ast
import json
from pathlib import Path

from scripts.build_vkr_report import append_summary


def test_notebook_cells_and_explicit_switch():
    notebook = json.loads(
        (Path(__file__).parents[1] / "audio_deepfake_detector" / "notebooks"
         / "07_external_audio_evaluation.ipynb").read_text(encoding="utf-8")
    )
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 7
    for source in cells:
        ast.parse(source)
    evaluation = cells[4]
    assert "RUN_EXTERNAL_EVALUATION = False" in evaluation
    assert "if RUN_EXTERNAL_EVALUATION:" in evaluation
    assert "evaluate_external(" in evaluation
    assert "train_cnn(" not in "\n".join(cells)
    assert "best_model.pth" in "\n".join(cells)


def test_external_report_append_is_separate_and_idempotent():
    original = "# Журнал\n\n## RuASD validation\nПрежний результат.\n"
    summary = "# Внешняя оценка baseline_cached_v1\n\nНовый результат.\n"
    updated = append_summary(original, summary)
    assert "Прежний результат" in updated
    assert "Внешняя оценка" in updated
    assert append_summary(updated, summary) == updated
