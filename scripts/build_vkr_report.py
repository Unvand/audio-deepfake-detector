"""Собрать Word-отчёт и графики из редактируемого журнала."""

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

DOCS = Path(__file__).resolve().parents[1] / "docs"
MARKER = "<!-- TRAINING_FIGURE -->"


def training_figure(markdown: str, output: Path) -> None:
    """Перенести значения из таблицы в графики, сохранив пропуски."""
    baseline = markdown.split("## Результаты baseline_cached_v1", 1)[1].split(
        MARKER, 1
    )[0]
    rows = []
    for line in baseline.splitlines():
        if line.startswith("|"):
            epoch, *values = (value.strip() for value in line.strip("|").split("|"))
            if epoch.isdecimal():
                rows.append(
                    (
                        int(epoch),
                        *(float("nan") if x == "—" else float(x) for x in values),
                    )
                )
    if not rows:
        raise ValueError("В таблице baseline_cached_v1 нет значений для графиков")
    epochs, train, validation, f1, eer = zip(*rows, strict=True)
    figure = Figure(figsize=(8, 6), layout="constrained")
    FigureCanvasAgg(figure)
    loss, f1_ax, eer_ax = figure.subplots(3, 1, sharex=True)
    loss.plot(epochs, train, "o-", label="Train")
    loss.plot(epochs, validation, "s-", label="Validation")
    loss.set_ylabel("Потери")
    loss.legend()
    f1_ax.plot(epochs, f1, "o-", color="#15605e")
    f1_ax.set_ylabel("Validation F1")
    eer_ax.plot(epochs, eer, "o-", color="#963d24")
    eer_ax.set(ylabel="Validation EER", xlabel="Эпоха", xticks=epochs)
    for axis in (loss, f1_ax, eer_ax):
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)


def append_summary(markdown: str, summary: str) -> str:
    """Добавить результат отдельной записью и исключить повторный импорт."""
    digest = hashlib.sha256(summary.strip().encode()).hexdigest()
    marker = f"<!-- evaluation:{digest} -->"
    if marker in markdown:
        return markdown
    lines = summary.strip().splitlines()
    if not lines or not lines[0].startswith(("# Независимая оценка ", "# Внешняя оценка ")):
        raise ValueError("Ожидается evaluation_summary.md из notebook 06 или 07")
    nested = "\n".join("#" + line if line.startswith("#") else line for line in lines)
    return markdown.rstrip() + f"\n\n{marker}\n{nested}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append-evaluation", type=Path)
    args = parser.parse_args()
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise SystemExit("Установите Pandoc для экспорта в DOCX")
    source = DOCS / "VKR_EXPERIMENT_LOG.md"
    markdown = source.read_text(encoding="utf-8")
    if args.append_evaluation:
        markdown = append_summary(
            markdown, args.append_evaluation.read_text(encoding="utf-8")
        )
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        figure = work / "baseline.png"
        training_figure(markdown, figure)
        report_source = work / "report.md"
        report_source.write_text(
            markdown.replace(
                MARKER,
                "![Динамика обучения baseline_cached_v1](baseline.png){width=15cm}",
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                pandoc,
                str(report_source),
                "--from=markdown",
                "--to=docx",
                "--metadata=lang:ru-RU",
                f"--resource-path={work}:{DOCS}",
                "--output",
                str(DOCS / "VKR_EXPERIMENT_REPORT.docx"),
            ],
            check=True,
        )
    if args.append_evaluation:
        source.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
