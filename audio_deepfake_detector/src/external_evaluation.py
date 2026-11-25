"""Независимая оценка сохранённой CNN на пользовательских аудиофайлах."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
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
from tqdm.auto import tqdm

from audio_deepfake_detector.src.cnn_training import RuASDSegmentDataset, _run_epoch
from audio_deepfake_detector.src.colab_cache import _sha256
from audio_deepfake_detector.src.ruasd_evaluation import _preprocessing, load_best_model
from audio_deepfake_detector.src.ruasd_preparation import (
    resample_mono,
    segment_with_padding,
)
from audio_deepfake_detector.src.training_metrics import binary_metrics

EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
LABELS = {"REAL": 0, "FAKE": 1}


def ensure_external_folders(root: Path) -> dict[str, Path]:
    """Создать только каталоги ввода и результатов на Drive."""
    root = Path(root)
    folders = {
        "REAL": root / "input" / "REAL",
        "FAKE": root / "input" / "FAKE",
        "results": root / "results",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """Декодировать файл в float32 и усреднить каналы как при подготовке RuASD.

    Для форматов, недоступных libsndfile, используется установленный ffmpeg.
    Исходный файл не изменяется.
    """
    path = Path(path)
    try:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except (RuntimeError, sf.LibsndfileError):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"Не удалось декодировать {path}; ffmpeg недоступен") from None
        with tempfile.TemporaryDirectory() as temporary:
            decoded = Path(temporary) / "decoded.wav"
            process = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                 "-f", "wav", "-c:a", "pcm_f32le", str(decoded)],
                capture_output=True, text=True, check=False,
            )
            if process.returncode:
                raise ValueError(f"Ошибка декодирования {path}: {process.stderr.strip()}")
            waveform, sample_rate = sf.read(decoded, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"Пустой или некорректный сигнал: {path}")
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def scan_external_files(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Найти файлы, проверить декодирование и запретить пересечение классов.

    Возвращает корректные файлы и таблицу ошибок. Хеш привязан к содержимому;
    ID записи привязан к классу и относительному пути.
    """
    root = Path(root).resolve()
    records: list[dict] = []
    errors: list[dict] = []
    hashes: dict[str, str] = {}
    for class_name, class_value in LABELS.items():
        folder = root / "input" / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Нет папки {folder}")
        candidates = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
        for path in tqdm(candidates, desc=f"Проверка {class_name}", unit="файл"):
            if path.is_symlink() or path.resolve() != path:
                errors.append({"path": str(path), "error": "Символическая ссылка недопустима"})
                continue
            relative = path.relative_to(root / "input").as_posix()
            try:
                digest = _sha256(path)
                previous = hashes.get(digest)
                if previous and previous.split("/", 1)[0] != class_name:
                    raise ValueError(
                        f"Одинаковое содержимое в REAL и FAKE: {previous}, {relative}"
                    )
                hashes[digest] = relative
                waveform, rate = decode_audio(path)
                records.append({
                    "source_file_id": hashlib.sha256(relative.encode()).hexdigest(),
                    "relative_path": relative,
                    "audio_path": str(path),
                    "true_class": class_name,
                    "true_label": class_value,
                    "sha256": digest,
                    "sample_rate": rate,
                    "duration_seconds": len(waveform) / rate,
                    "group": path.relative_to(folder).parts[0] if len(path.relative_to(folder).parts) > 1 else "(без подпапки)",
                })
            except ValueError as exc:
                if "Одинаковое содержимое в REAL и FAKE" in str(exc):
                    raise
                errors.append({"path": str(path), "error": str(exc)})
            except (OSError, RuntimeError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
    return pd.DataFrame(records), pd.DataFrame(errors, columns=["path", "error"])


def prepare_segments(files: pd.DataFrame, output: Path, *, sample_rate: int,
                     segment_seconds: float = 4.0, local_copy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Подготовить временные PCM_16 FLAC по тем же правилам, что и RuASD."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    for file in tqdm(files.itertuples(index=False), total=len(files), desc="Декодирование и подготовка", unit="файл"):
        source = Path(file.audio_path)
        try:
            if _sha256(source) != file.sha256:
                raise ValueError("Файл изменился после предварительной проверки")
            if local_copy:
                local = output / "input" / file.relative_path
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, local)
                if _sha256(local) != file.sha256:
                    raise ValueError("Локальная копия не совпадает с исходным файлом")
                source = local
            waveform, rate = decode_audio(source)
            mono = resample_mono(waveform, rate, sample_rate)
            for index, (segment, valid_seconds) in enumerate(
                segment_with_padding(mono, sample_rate, segment_seconds)
            ):
                name = f"segments/{file.source_file_id}_{index:04d}.flac"
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, segment, sample_rate, format="FLAC", subtype="PCM_16")
                rows.append({
                    "sample_id": file.source_file_id, "segment_id": f"{file.source_file_id}_{index:04d}",
                    "split": "test_unseen", "label": file.true_class.lower(),
                    "audio_path": name, "source_path": file.audio_path,
                    "relative_path": file.relative_path, "group": file.group,
                    "valid_seconds": valid_seconds,
                })
        except (OSError, RuntimeError, ValueError) as exc:
            rows = [row for row in rows if row["sample_id"] != file.source_file_id]
            failures.append({"path": file.audio_path, "error": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(failures, columns=["path", "error"])


def prediction_tables(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Усреднить FAKE-score строго внутри одного source_file_id."""
    if len(rows) != len(scores):
        raise ValueError("Количество сегментов и прогнозов различается")
    segments = rows.copy()
    segments["true_label"] = segments["label"].map({"real": 0, "fake": 1})
    segments["fake_score"] = np.asarray(scores, dtype=float)
    if segments["fake_score"].isna().any() or not segments["fake_score"].between(0, 1).all():
        raise ValueError("Недопустимый FAKE-score")
    files = segments.groupby("sample_id", sort=False).agg(
        relative_path=("relative_path", "first"), source_path=("source_path", "first"),
        true_label=("true_label", "first"), true_class=("label", "first"),
        group=("group", "first"), n_segments=("segment_id", "size"),
        fake_score=("fake_score", "mean"),
    ).reset_index().rename(columns={"sample_id": "source_file_id"})
    if segments.groupby("sample_id")["true_label"].nunique().gt(1).any():
        raise ValueError("У файла обнаружены разные истинные классы")
    segments["source_file_id"] = segments["sample_id"]
    for table in (segments, files):
        table["predicted_label"] = table["fake_score"].ge(threshold).astype(int)
        table["predicted_class"] = table["predicted_label"].map({0: "REAL", 1: "FAKE"})
    files["true_class"] = files["true_class"].str.upper()
    files["correct"] = files["true_label"].eq(files["predicted_label"])
    return segments, files


def safe_metrics(table: pd.DataFrame, threshold: float) -> dict:
    """Не подменять неопределимые метрики нулями при одном классе."""
    if table.empty:
        return {key: None for key in ("accuracy", "precision", "recall", "f1", "eer", "roc_auc", "far_fpr", "frr_fnr", "tn", "fp", "fn", "tp")}
    labels = table["true_label"].to_numpy(dtype=int)
    scores = table["fake_score"].to_numpy(dtype=float)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel())
    both = len(np.unique(labels)) == 2
    base = binary_metrics(labels, scores, threshold) if both else {}
    return {
        "accuracy": (tn + tp) / len(labels),
        "precision": base.get("precision"), "recall": base.get("recall"),
        "f1": base.get("f1"), "eer": base.get("eer"),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "far_fpr": fp / (fp + tn) if fp + tn else None,
        "frr_fnr": fn / (fn + tp) if fn + tp else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate_external(files: pd.DataFrame, checkpoint: Path, preprocessing_path: Path, *, batch_size: int = 32,
                      local_copy: bool = False, git_commit: str = "UNKNOWN") -> dict:
    """Оценить внешние файлы без optimizer; временные сегменты удалить после инференса."""
    if batch_size <= 0 or files.empty:
        raise ValueError("Нужны аудиофайлы и положительный batch size")
    checkpoint = Path(checkpoint)
    sha = _sha256(checkpoint)
    model, mel, threshold, saved = load_best_model(checkpoint)
    preprocessing = _preprocessing(preprocessing_path, mel)
    if preprocessing.get("flac_subtype") != "PCM_16":
        raise ValueError("FLAC subtype подготовки RuASD отличается от PCM_16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory(prefix="external_audio_", dir="/content" if Path("/content").is_dir() else None) as temporary:
        rows, errors = prepare_segments(files, Path(temporary), sample_rate=mel.sample_rate,
                                        local_copy=local_copy)
        if rows.empty:
            raise ValueError("Ни один аудиофайл не удалось подготовить")
        dataset = RuASDSegmentDataset(rows, temporary, "test_unseen", mel)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=device.type == "cuda")
        print(f"Инференс CNN: {len(rows)} сегментов, {len(loader)} batch, {device}")
        prediction = _run_epoch(model.to(device), loader, nn.BCEWithLogitsLoss(), device,
                                epoch=0, total_epochs=1, threshold=threshold)
        if prediction["sample_ids"] != rows["sample_id"].tolist():
            raise RuntimeError("Порядок прогнозов и сегментов различается")
        segments, sources = prediction_tables(rows, prediction["scores"], threshold)
    if _sha256(checkpoint) != sha:
        raise RuntimeError("Checkpoint изменился во время оценки")
    config = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(), "git_commit": git_commit,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
        "checkpoint_epoch": int(saved["epoch"]) + 1, "model": type(model).__name__,
        "mel_config": asdict(mel), "threshold": threshold,
        "preprocessing_config": preprocessing,
        "segmentation": "mono channel mean; scipy resample_poly; 4 seconds non-overlap; zero-pad final; PCM_16 FLAC",
        "aggregation": "mean sigmoid FAKE-score per source_file_id",
        "device": str(device), "batch_size": batch_size, "local_input_copy": local_copy,
        "input_count": len(files), "evaluated_count": len(sources),
        "input_sha256": dict(zip(files["relative_path"], files["sha256"], strict=True)),
    }
    group_metrics = (
        sources.groupby(["true_class", "group"], sort=True)
        .agg(n_files=("source_file_id", "size"), accuracy=("correct", "mean"),
             mean_fake_score=("fake_score", "mean"))
        .reset_index()
    )
    return {"segments": segments, "files": sources, "errors": errors,
            "group_metrics": group_metrics,
            "metrics": {"file": safe_metrics(sources, threshold),
                        "segment": safe_metrics(segments, threshold)}, "config": config}


def external_figures(result: dict) -> dict[str, plt.Figure]:
    """Построить графики по фактическим предсказаниям файлов."""
    files, metrics = result["files"], result["metrics"]["file"]
    threshold = result["config"]["threshold"]
    figures = {}
    fig, ax = plt.subplots(layout="constrained")
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["REAL", "FAKE"],
           yticklabels=["REAL", "FAKE"], xlabel="Прогноз", ylabel="Истинный класс",
           title="Матрица ошибок: внешние файлы")
    figures["confusion_matrix"] = fig
    fig, ax = plt.subplots(layout="constrained")
    for label, group in files.groupby("true_class"):
        ax.hist(group["fake_score"], bins=np.linspace(0, 1, 21), alpha=.5, label=label)
    ax.axvline(threshold, color="black", linestyle="--", label="Порог")
    ax.set(title="FAKE-score: REAL и FAKE", xlabel="FAKE-score файла", ylabel="Число файлов")
    ax.legend()
    figures["score_distribution"] = fig
    fig, ax = plt.subplots(layout="constrained")
    errors = files.loc[~files["correct"]]
    counts = errors["true_class"].value_counts().reindex(["REAL", "FAKE"], fill_value=0)
    ax.bar(["REAL → FAKE", "FAKE → REAL"], counts.to_numpy())
    ax.set(title="Ошибки классификации", ylabel="Число исходных файлов")
    figures["classification_errors"] = fig
    if metrics["roc_auc"] is not None:
        fig, ax = plt.subplots(layout="constrained")
        fpr, tpr, _ = roc_curve(files["true_label"], files["fake_score"])
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(title="ROC-кривая: внешние файлы", xlabel="FPR", ylabel="TPR")
        ax.legend()
        figures["roc_curve"] = fig
    groups = files.loc[files["group"].ne("(без подпапки)")]
    if not groups.empty:
        fig, ax = plt.subplots(layout="constrained")
        groups.groupby(["true_class", "group"])["correct"].mean().plot.bar(ax=ax)
        ax.set(title="Доля верных прогнозов по подпапкам", xlabel="Класс / подпапка", ylabel="Доля")
        figures["groups"] = fig
    return figures


def external_summary(result: dict) -> str:
    """Сформировать краткую запись журнала на основе реальных метрик."""
    metrics, cfg = result["metrics"]["file"], result["config"]
    def value(key: str) -> str:
        number = metrics[key]
        return "N/A" if number is None else f"{number:.4f}"
    return "\n".join([
        f"# Внешняя оценка {Path(cfg['checkpoint']).parent.name}", "",
        f"Дата UTC: {cfg['evaluated_at_utc']}. Git commit: `{cfg['git_commit']}`.",
        f"Checkpoint: `{cfg['checkpoint']}`; SHA-256: `{cfg['checkpoint_sha256']}`.",
        (f"Оценено {cfg['evaluated_count']} независимых файлов из {cfg['input_count']}; "
         f"порог {cfg['threshold']}; положительный класс FAKE."), "",
        "## Метрики по исходным файлам", "",
        "| Accuracy | Precision | Recall | F1 | EER | ROC-AUC | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (f"| {value('accuracy')} | {value('precision')} | {value('recall')} | "
         f"{value('f1')} | {value('eer')} | {value('roc_auc')} | {metrics['fp']} | {metrics['fn']} |"), "",
        "## Ограничения", "",
        ("Это внешняя пользовательская выборка. Её размер и состав ограничивают вывод об обобщении; "
         "низкие результаты сохраняются без подбора порога по тесту. "
         "Одна или две записи диктора не характеризуют всех дикторов."), "",
    ])


def save_external(result: dict, figures: dict[str, plt.Figure], results_root: Path) -> Path:
    """Сохранить каждый запуск в новом каталоге без перезаписи результатов."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output = results_root / f"run_{stamp}_{uuid4().hex[:8]}"
    output.mkdir()
    files, segments = result["files"], result["segments"]
    files.to_csv(output / "file_predictions.csv", index=False)
    segments.to_csv(output / "segment_predictions.csv", index=False)
    result["group_metrics"].to_csv(output / "metrics_by_group.csv", index=False)
    for name, actual, predicted in (("false_positives", 0, 1), ("false_negatives", 1, 0)):
        files.loc[files["true_label"].eq(actual) & files["predicted_label"].eq(predicted)].to_csv(
            output / f"{name}.csv", index=False)
    result["errors"].to_csv(output / "decode_errors.csv", index=False)
    for name, data in (("metrics", result["metrics"]), ("evaluation_config", result["config"])):
        (output / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    (output / "evaluation_summary.md").write_text(external_summary(result), encoding="utf-8")
    return output
