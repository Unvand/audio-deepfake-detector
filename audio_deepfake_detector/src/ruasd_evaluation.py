"""Воспроизводимая оценка сохранённой CNN на test_unseen."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from audio_deepfake_detector.src.cnn_training import (
    AudioDeepfakeCNN,
    MelConfig,
    RuASDSegmentDataset,
    _run_epoch,
)
from audio_deepfake_detector.src.colab_cache import MANIFEST_NAME, _sha256
from audio_deepfake_detector.src.training_metrics import (
    aggregate_source_scores,
    binary_metrics,
)

TEST_SPLIT = "test_unseen"
MAPPING = {"real": 0, "fake": 1}
ERROR_COLUMNS = {"false_positives": (0, 1), "false_negatives": (1, 0)}


def read_test_manifest(prepared_root: Path) -> pd.DataFrame:
    """Проверить изоляцию test по sample_id и корректность путей."""
    full = pd.read_csv(
        prepared_root / MANIFEST_NAME,
        low_memory=False,
        dtype={"sample_id": str, "segment_id": str},
    )
    required = {"sample_id", "segment_id", "split", "label", "subset", "audio_path"}
    if not required <= set(full.columns):
        raise ValueError(
            f"Не хватает полей manifest: {sorted(required - set(full.columns))}"
        )
    test = full.loc[full["split"].eq(TEST_SPLIT)].copy()
    if test.empty or test[list(required)].isna().any().any():
        raise ValueError("test_unseen пуст или содержит пропущенные поля")
    if (
        test["sample_id"]
        .isin(full.loc[full["split"].ne(TEST_SPLIT), "sample_id"])
        .any()
    ):
        raise ValueError("Найдены общие sample_id между test_unseen и другими split")
    if test["segment_id"].duplicated().any() or test["audio_path"].duplicated().any():
        raise ValueError("Дубли segment_id или audio_path в test_unseen")
    test["label"] = test["label"].str.lower()
    if set(test["label"]) != set(MAPPING):
        raise ValueError("В test_unseen нужны оба класса: REAL и FAKE")
    if test.groupby("sample_id")[["label", "subset"]].nunique().gt(1).any().any():
        raise ValueError("У одного sample_id расходятся класс или источник")
    for name in test["audio_path"]:
        path = Path(name)
        if (
            path.is_absolute()
            or len(path.parts) < 2
            or path.parts[0] != TEST_SPLIT
            or ".." in path.parts
            or "\\" in name
            or path.suffix.lower() != ".flac"
        ):
            raise ValueError(f"Недопустимый путь test_unseen: {name}")
    return test.reset_index(drop=True)


def load_best_model(checkpoint: Path, *, expected_epoch: int | None = None):
    """Загрузить доверенный checkpoint с Drive и строго сверить веса и конфигурацию."""
    checkpoint = Path(checkpoint)
    if checkpoint.name != "best_model.pth":
        raise ValueError("Для независимой оценки требуется best_model.pth")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("class_mapping") != MAPPING:
        raise ValueError("Checkpoint должен использовать REAL=0 и FAKE=1")
    mel = MelConfig(**saved["mel_config"])
    mel.validate()
    threshold = float(saved["training_config"]["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Недопустимый сохранённый порог")
    epoch = int(saved["epoch"]) + 1
    if epoch != int(saved["best_epoch"]) + 1 or (
        expected_epoch is not None and epoch != expected_epoch
    ):
        raise ValueError(f"Неподходящая лучшая эпоха в {checkpoint}: {epoch}")
    model = AudioDeepfakeCNN()
    model.load_state_dict(saved["model_state"], strict=True)
    return model.eval().requires_grad_(False), mel, threshold, saved


def _preprocessing(path, mel):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    seconds = float(config["segment_seconds"])
    if (
        config["target_sample_rate"] != mel.sample_rate
        or config["output_format"].upper() != "FLAC"
        or seconds != 4.0
    ):
        raise ValueError(
            "Конфигурация подготовки не соответствует checkpoint и сегментам FLAC 4 с"
        )
    return config


def check_test_files(root, rows, config):
    """Проверить наличие и параметры FLAC строго в пределах тестового split."""
    root = Path(root).resolve()
    for name in rows["audio_path"]:
        path = root / name
        if path.resolve() != path:
            raise ValueError(
                f"Путь выходит за пределы тестового каталога или содержит ссылку: {name}"
            )
        info = sf.info(path)
        if (
            info.format != "FLAC"
            or info.samplerate != config["target_sample_rate"]
            or info.frames != round(config["segment_seconds"] * info.samplerate)
        ):
            raise ValueError(f"Параметры файла не соответствуют подготовке: {name}")


def prediction_tables(rows, scores, threshold):
    """Агрегировать равновзвешенные sigmoid-оценки внутри sample_id."""
    segments = rows[["segment_id", "sample_id", "subset", "label", "audio_path"]].copy()
    segments["true_label"] = segments.pop("label").map(MAPPING)
    segments["fake_score"] = np.asarray(scores, dtype=float)
    if not segments["fake_score"].between(0, 1).all():
        raise ValueError("Некорректные FAKE-score")
    groups = segments.groupby("sample_id", sort=False)
    source = groups.agg(
        subset=("subset", "first"),
        n_segments=("segment_id", "size"),
        audio_path=("audio_path", "first"),
    ).reset_index()
    source["true_label"], source["fake_score"] = aggregate_source_scores(
        segments["sample_id"].tolist(),
        segments["true_label"].to_numpy(),
        segments["fake_score"].to_numpy(),
    )
    for table in (segments, source):
        table["predicted_label"] = table["fake_score"].ge(threshold).astype(int)
        table["true_class"] = table["true_label"].map({0: "REAL", 1: "FAKE"})
        table["predicted_class"] = table["predicted_label"].map({0: "REAL", 1: "FAKE"})
    source["correct"] = source["true_label"].eq(source["predicted_label"])
    return segments, source


def evaluation_metrics(table, threshold):
    labels, scores = table["true_label"].to_numpy(), table["fake_score"].to_numpy()
    matrix = confusion_matrix(labels, scores >= threshold, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {
        **binary_metrics(labels, scores, threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "far_fpr": fp / (fp + tn),
        "frr_fnr": fn / (fn + tp),
        "n": len(table),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_unseen(
    prepared_root,
    checkpoint,
    preprocessing_path,
    *,
    batch_size=32,
    expected_epoch=None,
    device=None,
    git_commit,
    run_id,
):
    """Оценить только test_unseen на CPU или CUDA без обновления модели."""
    root, checkpoint = Path(prepared_root), Path(checkpoint)
    if batch_size <= 0:
        raise ValueError("Batch size должен быть положительным")
    checkpoint_sha = _sha256(checkpoint)
    model, mel, threshold, saved = load_best_model(
        checkpoint, expected_epoch=expected_epoch
    )
    config = _preprocessing(preprocessing_path, mel)
    rows = read_test_manifest(root)
    check_test_files(root, rows, config)
    dataset = RuASDSegmentDataset(rows, root, TEST_SPLIT, mel)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda"
    )
    result = _run_epoch(
        model.to(device),
        loader,
        nn.BCEWithLogitsLoss(),
        device,
        epoch=0,
        total_epochs=1,
        threshold=threshold,
    )
    if (
        result["sample_ids"] != rows["sample_id"].tolist()
        or _sha256(checkpoint) != checkpoint_sha
    ):
        raise RuntimeError(
            "Порядок предсказаний или checkpoint изменился во время оценки"
        )
    segments, source = prediction_tables(rows, result["scores"], threshold)
    metrics = {
        "segment": evaluation_metrics(segments, threshold),
        "source": evaluation_metrics(source, threshold),
    }
    subsets = (
        source.groupby(["subset", "true_class"], sort=True)
        .agg(
            n_records=("sample_id", "size"),
            accuracy=("correct", "mean"),
            mean_fake_score=("fake_score", "mean"),
        )
        .reset_index()
    )
    tables = {
        "segment_predictions": segments,
        "source_predictions": source,
        "metrics_by_subset": subsets,
    }
    for name, (actual, prediction) in ERROR_COLUMNS.items():
        tables[name] = source.loc[
            source["true_label"].eq(actual) & source["predicted_label"].eq(prediction)
        ]
    epoch_record = next(
        (row for row in saved.get("history", []) if row["epoch"] == saved["epoch"]),
        {},
    )
    comparison = pd.DataFrame(
        {
            "validation": epoch_record.get("validation_source", {}),
            TEST_SPLIT: metrics["source"],
        }
    ).reindex(
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "eer",
            "roc_auc",
            "far_fpr",
            "frr_fnr",
        ]
    )
    return {
        "metrics": metrics,
        "tables": tables,
        "comparison": comparison,
        "config": {
            "run_id": run_id,
            "split": TEST_SPLIT,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "model": type(model).__name__,
            "epoch": int(saved["epoch"]) + 1,
            "threshold": threshold,
            "class_mapping": MAPPING,
            "mel_config": asdict(mel),
            "training_config": saved["training_config"],
            "preprocessing": config,
            "preprocessing_path": str(preprocessing_path),
            "git_commit": git_commit,
            "prepared_root": str(root),
            "manifest_sha256": _sha256(root / MANIFEST_NAME),
            "device": str(device),
            "batch_size": batch_size,
            "inference_precision": "float32",
            "aggregation": "mean sigmoid score per sample_id",
            "eer_method": "nearest ROC point to FPR=FNR, as in validation",
        },
    }


def evaluation_figures(result):
    """Графики матриц, ROC и распределений без повторного инференса."""
    figures = {}
    threshold = result["config"]["threshold"]
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    roc, roc_ax = plt.subplots(layout="constrained")
    for ax, (level, table_name, title) in zip(
        axes,
        (
            ("source", "source_predictions", "Исходные записи"),
            ("segment", "segment_predictions", "Сегменты"),
        ),
        strict=True,
    ):
        table = result["tables"][table_name]
        stats = result["metrics"][level]
        matrix = np.array([[stats["tn"], stats["fp"]], [stats["fn"], stats["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"],
            title=title,
            xlabel="Прогноз",
            ylabel="Истинный класс",
        )
        fpr, tpr, _ = roc_curve(table["true_label"], table["fake_score"])
        roc_ax.plot(fpr, tpr, label=f"{title} · AUC {stats['roc_auc']:.4f}")
    figure.suptitle("Матрицы ошибок на test_unseen")
    figures["confusion_matrix"] = figure
    roc_ax.plot([0, 1], [0, 1], "--", color="gray")
    roc_ax.set(
        title="ROC-кривая на test_unseen",
        xlabel="Доля ложных тревог (FPR)",
        ylabel="Доля обнаруженных FAKE (TPR)",
    )
    roc_ax.legend()
    figures["roc_curve"] = roc

    sources = result["tables"]["source_predictions"]
    for name, key, title in (
        ("score_distribution", "true_class", "Распределение FAKE-score по классам"),
        ("score_by_subset", "subset", "Распределение FAKE-score по источникам"),
    ):
        figure, ax = plt.subplots(layout="constrained")
        for group, data in sources.groupby(key):
            ax.hist(
                data["fake_score"],
                bins=np.linspace(0, 1, 21),
                density=True,
                histtype="step",
                linewidth=1.5,
                label=group,
            )
        ax.axvline(threshold, linestyle="--", color="gray", label="Порог")
        ax.set(title=title, xlabel="Средний FAKE-score записи", ylabel="Плотность")
        ax.legend()
        figures[name] = figure
    figure, ax = plt.subplots(layout="constrained")
    result["comparison"].loc[["accuracy", "precision", "recall", "f1", "eer"]].plot.bar(
        ax=ax
    )
    ax.set(
        title="Validation и test_unseen на уровне записей",
        xlabel="Метрика",
        ylabel="Значение",
    )
    ax.tick_params(axis="x", labelrotation=0)
    figures["validation_vs_test"] = figure
    return figures


def evaluation_summary(result):
    """Сформировать текст для ВКР на основе фактической оценки."""
    cfg, stats = result["config"], result["metrics"]["source"]
    subsets = result["tables"]["metrics_by_subset"]
    text = [
        f"# Независимая оценка {cfg['run_id']}",
        "",
        f"Модель — `{Path(cfg['checkpoint']).name}`, эпоха {cfg['epoch']}, код `{cfg['git_commit']}`.",
        (
            f"Выборка — RuASD test_unseen, {stats['n']} записей и "
            f"{result['metrics']['segment']['n']} сегментов. Порог из checkpoint — {cfg['threshold']}."
        ),
        "Положительный класс — FAKE. Основные метрики рассчитаны по исходным записям.",
        "",
        "## Результаты",
        "",
        (
            f"Accuracy {stats['accuracy']:.4f}, Precision {stats['precision']:.4f}, "
            f"Recall {stats['recall']:.4f}, F1 {stats['f1']:.4f}, EER {stats['eer']:.4f}, "
            f"ROC-AUC {stats['roc_auc']:.4f}, FAR {stats['far_fpr']:.4f}, FRR {stats['frr_fnr']:.4f}."
        ),
        "",
        "## Источники",
        "",
        "| Источник | Класс | Записи | Доля верных | Средний FAKE-score |",
        "|---|---|---:|---:|---:|",
    ]
    for row in subsets.itertuples(index=False):
        text.append(
            f"| {row.subset} | {row.true_class} | {row.n_records} | "
            f"{row.accuracy:.4f} | {row.mean_fake_score:.4f} |"
        )
    text += [
        "",
        "## Ошибки",
        "",
        f"Ложные тревоги (REAL → FAKE) — {stats['fp']}. Пропуски (FAKE → REAL) — {stats['fn']}.",
    ]
    for table_name in ERROR_COLUMNS:
        errors = result["tables"][table_name]
        if len(errors):
            counts = errors["subset"].value_counts()
            text.append(
                f"{table_name} по источникам — "
                + ", ".join(f"{name}: {count}" for name, count in counts.items())
                + "."
            )
    generators = subsets.loc[subsets["true_class"].eq("FAKE")]
    rates = ", ".join(
        f"{row.subset} {row.accuracy:.1%}" for row in generators.itertuples(index=False)
    )
    text += [
        "",
        "## Обобщение и ограничения",
        "",
        f"Доля обнаружения на представленных неизвестных генераторах — {rates}.",
        (
            "Вывод ограничен генераторами и корпусами этого теста. На оценку могут влиять источник, "
            "кодирование и заполнение тишиной. Разделение по дикторам не подтверждено."
        ),
        (
            "Для оценки переноса на RVC нужна внешняя выборка. По спектрограмме отдельной ошибки "
            "нельзя определить её причину."
        ),
        (
            "Если результаты этого теста используются для доработки модели или порога, "
            "для следующего финального вывода требуется новый независимый тест."
        ),
        "",
    ]
    return "\n".join(text)


def save_evaluation(result, figures, output_dir):
    """Сохранить новый отчёт. Повторная оценка получает отдельный подкаталог."""
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = output / f"repeat-{stamp}-{uuid4().hex[:8]}"
        output.mkdir()
    for name, payload in (
        ("test_metrics", result["metrics"]),
        ("evaluation_config", result["config"]),
    ):
        (output / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    pd.DataFrame(result["metrics"]).T.rename_axis("level").to_csv(
        output / "test_metrics.csv"
    )
    for name, table in result["tables"].items():
        table.to_csv(output / f"{name}.csv", index=False)
    result["comparison"].rename_axis("metric").to_csv(output / "validation_vs_test.csv")
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    (output / "evaluation_summary.md").write_text(
        evaluation_summary(result), encoding="utf-8"
    )
    return output
