"""Обучение одной CNN на log-Mel представлениях подготовленного RuASD.

Модуль не связан с GUI и не запускает обучение при импорте. ``test_unseen``
исключён из loaders обучения и вызывается только отдельной функцией оценки.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm.auto import tqdm

from audio_deepfake_detector.src.training_metrics import (
    aggregate_source_scores,
    binary_metrics,
)


@dataclass(frozen=True)
class MelConfig:
    """Явно зафиксированные параметры log-Mel-преобразования."""

    sample_rate: int = 32_000
    n_fft: int = 1_024
    win_length: int = 1_024
    hop_length: int = 320
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 16_000.0
    power: float = 2.0

    def validate(self) -> None:
        """Проверить физически и вычислительно допустимые параметры."""
        if self.sample_rate != 32_000:
            raise ValueError("RuASD подготовлен с фиксированной частотой 32 000 Гц")
        if self.n_fft <= 0 or self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("FFT, длина окна и hop_length должны быть положительными")
        if self.win_length > self.n_fft:
            raise ValueError("win_length не должен превышать n_fft")
        if self.n_mels < 16 or (4 * self.sample_rate // self.hop_length + 1) < 16:
            raise ValueError("Для четырёх слоёв pooling требуется минимум 16 Mel-частот и кадров")
        if self.n_fft // 2 >= 4 * self.sample_rate:
            raise ValueError("FFT слишком велик для четырёхсекундного сегмента")
        if not all(map(isfinite, (self.f_min, self.f_max))) or not 0 <= self.f_min < self.f_max:
            raise ValueError("Требуется 0 <= f_min < f_max")
        if self.f_max > self.sample_rate / 2:
            raise ValueError("f_max не должен превышать частоту Найквиста")


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры одного воспроизводимого запуска."""

    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    threshold: float = 0.5
    early_stopping_patience: int = 5
    balance_classes: bool = True
    use_amp: bool = True
    confusion_matrix_interval: int = 1
    spectrogram_examples_per_class: int = 4

    def validate(self) -> None:
        """Проверить настройки до создания loaders и checkpoint."""
        if self.batch_size not in {8, 16, 32, 64}:
            raise ValueError("batch_size должен быть одним из: 8, 16, 32, 64")
        if (
            not all(map(isfinite, (self.learning_rate, self.weight_decay)))
            or self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0
        ):
            raise ValueError("epochs/lr должны быть положительными, weight_decay >= 0")
        if self.num_workers < 0 or self.early_stopping_patience <= 0:
            raise ValueError("num_workers >= 0, patience > 0")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold должен находиться между 0 и 1")
        if self.confusion_matrix_interval <= 0:
            raise ValueError("Интервал confusion matrix должен быть положительным")
        if not 3 <= self.spectrogram_examples_per_class <= 5:
            raise ValueError("Показывайте от 3 до 5 Mel-примеров каждого класса")


def configs_from_mapping(values: Mapping[str, Any]) -> tuple[MelConfig, TrainingConfig]:
    """Создать реальные конфигурации из значений панели Colab."""
    mel = MelConfig(
        sample_rate=32_000,
        n_fft=int(values["n_fft"]),
        win_length=int(values["win_length"]),
        hop_length=int(values["hop_length"]),
        n_mels=int(values["n_mels"]),
        f_min=float(values["f_min"]),
        f_max=float(values["f_max"]),
    )
    training = TrainingConfig(
        batch_size=int(values["batch_size"]),
        epochs=int(values["epochs"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        num_workers=int(values["num_workers"]),
        seed=int(values["seed"]),
        balance_classes=bool(values["balance_classes"]),
        use_amp=bool(values["use_amp"]),
    )
    mel.validate()
    training.validate()
    return mel, training


def resolve_run_directory(training_root: str | Path, run_id: str) -> Path:
    """Вернуть отдельный безопасный каталог эксперимента."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("run_id: 1–64 символа A-Z, a-z, 0-9, точка, '_' или '-'")
    return Path(training_root) / "runs" / run_id


class RuASDSegmentDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Читать FLAC-сегменты и вычислять log-Mel на лету."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        prepared_root: str | Path,
        split: str,
        mel_config: MelConfig | None = None,
    ) -> None:
        if split not in {"train", "validation", "test_unseen"}:
            raise ValueError(f"Неизвестный split: {split}")
        self.split = split
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Нет сегментов для split={split}")
        if self.rows["segment_id"].duplicated().any():
            raise ValueError("segment_id должны быть уникальны")
        self.prepared_root = Path(prepared_root)
        self.mel_config = mel_config or MelConfig()
        self.mel_config.validate()
        config = self.mel_config
        self.to_mel = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
        )
        self.to_db = AmplitudeToDB(stype="power", top_db=80.0)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_log_mel(self, index: int) -> tuple[torch.Tensor, pd.Series]:
        row = self.rows.iloc[index]
        path = self.prepared_root / str(row["audio_path"])
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != self.mel_config.sample_rate:
            raise ValueError(
                f"{path}: ожидалось {self.mel_config.sample_rate} Гц, получено {sample_rate}"
            )
        if not np.isfinite(waveform).all():
            raise ValueError(f"{path}: сигнал содержит NaN/Inf")
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        log_mel_db = self.to_db(self.to_mel(tensor))
        if not torch.isfinite(log_mel_db).all():
            raise ValueError(f"{path}: log-Mel содержит NaN/Inf")
        return log_mel_db, row

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        log_mel_db, row = self._load_log_mel(index)
        standardized = (log_mel_db - log_mel_db.mean()) / log_mel_db.std().clamp_min(
            1e-6
        )
        label = torch.tensor(float(str(row["label"]).lower() == "fake"))
        return standardized.unsqueeze(0), label, str(row["sample_id"])

    def visualization_item(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        """Вернуть физически интерпретируемый log-Mel в дБ до стандартизации."""
        log_mel_db, row = self._load_log_mel(index)
        metadata = {
            "label": str(row["label"]).upper(),
            "subset": str(row["subset"]),
            "segment_id": str(row["segment_id"]),
            "sample_id": str(row["sample_id"]),
        }
        return log_mel_db, metadata


class AudioDeepfakeCNN(nn.Module):
    """Одна компактная 2D-CNN для бинарной классификации log-Mel."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(192, 1),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(log_mel)).squeeze(1)


def set_reproducible_seed(seed: int) -> None:
    """Зафиксировать доступные генераторы случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(manifest: pd.DataFrame) -> float:
    """Вычислить REAL/FAKE строго по фактическим train-сегментам."""
    train_labels = manifest.loc[manifest["split"] == "train", "label"].str.lower()
    real_count = int((train_labels == "real").sum())
    fake_count = int((train_labels == "fake").sum())
    if real_count == 0 or fake_count == 0:
        raise ValueError("Для pos_weight нужны оба класса в train")
    return real_count / fake_count


def build_loss_functions(
    manifest: pd.DataFrame,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, float, float]:
    """Создать взвешенный train loss и всегда невзвешенный validation loss."""
    computed_weight = compute_pos_weight(manifest)
    applied_weight = computed_weight if training_config.balance_classes else 1.0
    pos_weight = torch.tensor([applied_weight], dtype=torch.float32, device=device)
    return (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        nn.BCEWithLogitsLoss(),
        applied_weight,
        computed_weight,
    )


def make_loaders(
    manifest: pd.DataFrame,
    prepared_root: str | Path,
    mel_config: MelConfig,
    training_config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Создать loaders только для train и validation — без test_unseen."""
    train_dataset = RuASDSegmentDataset(manifest, prepared_root, "train", mel_config)
    validation_dataset = RuASDSegmentDataset(
        manifest, prepared_root, "validation", mel_config
    )
    common = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": training_config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels: list[float] = []
    scores: list[float] = []
    sample_ids: list[str] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.inference_mode()
    description = f"Epoch {epoch + 1}/{total_epochs} {'train' if training else 'validation'}"
    with context:
        progress = tqdm(loader, desc=description, unit="batch", leave=False)
        for batch_index, (log_mel, target, batch_ids) in enumerate(progress, start=1):
            log_mel = log_mel.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(log_mel)
                loss = loss_function(logits, target)
            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            labels.extend(target.detach().cpu().numpy().tolist())
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            sample_ids.extend(batch_ids)
            learning_rate = optimizer.param_groups[0]["lr"] if training else 0.0
            progress.set_postfix(
                batch=f"{batch_index}/{len(loader)}",
                loss=f"{loss_value:.4f}",
                lr=f"{learning_rate:.2e}" if training else "-",
            )
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    source_labels, source_scores = aggregate_source_scores(
        sample_ids, labels_array, scores_array
    )
    return {
        "loss": float(np.mean(losses)),
        "segment": binary_metrics(labels_array, scores_array, threshold),
        "source": binary_metrics(source_labels, source_scores, threshold),
        "labels": labels_array,
        "scores": scores_array,
        "sample_ids": sample_ids,
        "source_labels": source_labels,
        "source_scores": source_scores,
    }


def _spectrogram_indices(dataset: RuASDSegmentDataset, per_class: int) -> list[int]:
    labels = dataset.rows["label"].str.lower()
    real = dataset.rows.index[labels == "real"].tolist()[:per_class]
    fake = dataset.rows.index[labels == "fake"].tolist()[:per_class]
    if len(real) < per_class or len(fake) < per_class:
        raise ValueError("Недостаточно REAL/FAKE-примеров для TensorBoard")
    return real + fake


def create_spectrogram_figures(
    dataset: RuASDSegmentDataset,
    examples_per_class: int = 4,
) -> list[tuple[str, plt.Figure]]:
    """Построить фиксированные реальные log-Mel в дБ с общей цветовой шкалой."""
    indices = _spectrogram_indices(dataset, examples_per_class)
    items = [dataset.visualization_item(index) for index in indices]
    minimum = min(float(mel.min()) for mel, _ in items)
    maximum = max(float(mel.max()) for mel, _ in items)
    config = dataset.mel_config
    mel_min = 2595.0 * np.log10(1.0 + config.f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + config.f_max / 700.0)
    mel_centers = np.linspace(mel_min, mel_max, config.n_mels + 2)[1:-1]
    tick_indices = np.linspace(0, config.n_mels - 1, 5).round().astype(int)
    tick_bins = tick_indices.astype(float)
    tick_mels = mel_centers[tick_indices]
    tick_hz = 700.0 * (10 ** (tick_mels / 2595.0) - 1.0)
    figures: list[tuple[str, plt.Figure]] = []
    for mel, metadata in items:
        duration = (mel.shape[-1] - 1) * config.hop_length / config.sample_rate
        figure, axis = plt.subplots(figsize=(8, 3))
        image = axis.imshow(
            mel.cpu().numpy(),
            origin="lower",
            aspect="auto",
            extent=[0, duration, 0, config.n_mels - 1],
            vmin=minimum,
            vmax=maximum,
            cmap="magma",
        )
        axis.set_xlabel("Время, с")
        axis.set_ylabel("Частота, Гц (Mel-шкала)")
        axis.set_yticks(tick_bins, [f"{frequency:.0f}" for frequency in tick_hz])
        axis.set_title(
            f"{metadata['label']} | {metadata['subset']} | {metadata['segment_id']}"
        )
        figure.colorbar(image, ax=axis, label="дБ")
        figure.tight_layout()
        tag = f"Spectrograms/{metadata['label']}/{metadata['subset']}_{metadata['segment_id']}"
        figures.append((tag, figure))
    return figures


def log_fixed_spectrograms(
    writer: SummaryWriter,
    dataset: RuASDSegmentDataset,
    examples_per_class: int = 4,
    global_step: int = 0,
) -> None:
    """Записать небольшую фиксированную подборку реальных Mel в TensorBoard."""
    for tag, figure in create_spectrogram_figures(dataset, examples_per_class):
        writer.add_figure(tag, figure, global_step=global_step, close=True)
    writer.flush()


def _log_epoch_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    train_result: dict[str, Any],
    validation_result: dict[str, Any],
    learning_rate: float,
    duration_seconds: float,
    pos_weight: float,
    confusion_matrix_interval: int,
) -> None:
    writer.add_scalars(
        "Loss",
        {"train": train_result["loss"], "validation": validation_result["loss"]},
        epoch,
    )
    for metric, group in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("eer", "EER"),
    ]:
        writer.add_scalars(
            group,
            {
                "train_source": train_result["source"][metric],
                "validation_source": validation_result["source"][metric],
                "train_segment": train_result["segment"][metric],
                "validation_segment": validation_result["segment"][metric],
            },
            epoch,
        )
    writer.add_scalar("LearningRate/value", learning_rate, epoch)
    writer.add_scalar("Timing/epoch_seconds", duration_seconds, epoch)
    writer.add_scalar("Experiment/pos_weight", pos_weight, epoch)
    if (epoch + 1) % confusion_matrix_interval == 0:
        labels = validation_result["source_labels"]
        predictions = (validation_result["source_scores"] >= 0.5).astype(int)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(matrix, display_labels=["REAL", "FAKE"]).plot(
            ax=axis, cmap="Blues", colorbar=False
        )
        axis.set_title("Validation — уровень sample_id")
        figure.tight_layout()
        writer.add_figure("ConfusionMatrix/validation_source", figure, epoch, close=True)
    writer.flush()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _compatible_training_config(saved: Mapping[str, Any], current: TrainingConfig) -> bool:
    # epochs разрешено только увеличивать для продолжения завершённого запуска.
    current_values = asdict(current)
    saved_values = dict(saved)
    current_epochs = int(current_values.pop("epochs"))
    saved_epochs = saved_values.pop("epochs", None)
    if not isinstance(saved_epochs, int):
        return False
    return current_values == saved_values and current_epochs >= saved_epochs


def assert_checkpoint_compatible(
    checkpoint: Mapping[str, Any],
    mel_config: MelConfig,
    training_config: TrainingConfig,
) -> None:
    """Запретить незаметное продолжение эксперимента с другими параметрами."""
    problems: list[str] = []
    if checkpoint.get("mel_config") != asdict(mel_config):
        problems.append("MelConfig отличается")
    saved_training = checkpoint.get("training_config", {})
    if not _compatible_training_config(saved_training, training_config):
        problems.append("TrainingConfig отличается (разрешено только увеличить epochs)")
    if problems:
        raise ValueError(
            "Checkpoint несовместим: " + "; ".join(problems) + ". Используйте новый run_id."
        )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    _atomic_json(history, output_dir / "training_history.json")
    flattened = pd.json_normalize(history)
    csv_temporary = output_dir / "training_history.csv.tmp"
    flattened.to_csv(csv_temporary, index=False)
    csv_temporary.replace(output_dir / "training_history.csv")


def inspect_training_run(
    output_dir: str | Path,
    mel_config: MelConfig,
    training_config: TrainingConfig,
    *,
    resume: bool,
) -> int:
    """Проверить запуск из notebook 05 и вернуть номер следующей эпохи (с единицы)."""
    output = Path(output_dir)
    if not resume:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Этот run_id уже занят. Включите Resume или укажите новый run_id.")
        return 1

    last_path = output / "last_checkpoint.pth"
    if not last_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint отсутствует: {last_path}. Для нового запуска укажите свободный "
            "run_id и выключите Resume training."
        )
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    assert_checkpoint_compatible(checkpoint, mel_config, training_config)
    return int(checkpoint["epoch"]) + 2


def train_cnn(
    prepared_root: str | Path,
    output_dir: str | Path,
    mel_config: MelConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    resume: bool = True,
) -> tuple[AudioDeepfakeCNN, list[dict[str, Any]]]:
    """Обучить CNN с TensorBoard и checkpoint после каждой полной эпохи."""
    prepared_root = Path(prepared_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(prepared_root / "ruasd_segments_manifest.csv", low_memory=False)
    if set(manifest["split"].unique()) - {"train", "validation", "test_unseen"}:
        raise ValueError("Manifest содержит неизвестные split")
    mel_config = mel_config or MelConfig()
    training_config = training_config or TrainingConfig()
    mel_config.validate()
    training_config.validate()
    set_reproducible_seed(training_config.seed)
    train_loader, validation_loader = make_loaders(
        manifest, prepared_root, mel_config, training_config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loss, validation_loss, pos_weight, computed_pos_weight = build_loss_functions(
        manifest, training_config, device
    )
    amp_enabled = bool(training_config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    last_path = output_dir / "last_checkpoint.pth"
    best_path = output_dir / "best_model.pth"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    if resume and last_path.exists():
        # Состояния CPU RNG должны остаться на CPU и при запуске на CUDA.
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        assert_checkpoint_compatible(checkpoint, mel_config, training_config)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_epoch = int(checkpoint.get("best_epoch", -1))
        history = checkpoint.get("history", [])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
    elif not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            "Каталог run_id уже содержит файлы. Выберите новый run_id или включите resume."
        )

    if start_epoch >= training_config.epochs:
        tqdm.write("Все запрошенные эпохи уже завершены; checkpoint не изменён.")
        return model, history
    if epochs_without_improvement >= training_config.early_stopping_patience:
        tqdm.write("Early stopping уже сработал; для нового эксперимента смените run_id.")
        return model, history

    experiment_config = {
        "mel_config": asdict(mel_config),
        "training_config": asdict(training_config),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "pos_weight": pos_weight,
        "computed_pos_weight": computed_pos_weight,
        "class_mapping": {"real": 0, "fake": 1},
    }
    _atomic_json(experiment_config, output_dir / "experiment_config.json")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"), purge_step=start_epoch)
    writer.add_text(
        "Experiment/configuration",
        "```json\n" + json.dumps(experiment_config, ensure_ascii=False, indent=2) + "\n```",
        start_epoch,
    )
    if start_epoch == 0:
        log_fixed_spectrograms(
            writer,
            train_loader.dataset,
            training_config.spectrogram_examples_per_class,
        )

    try:
        for epoch in range(start_epoch, training_config.epochs):
            epoch_started = time.perf_counter()
            train_result = _run_epoch(
                model,
                train_loader,
                train_loss,
                device,
                epoch=epoch,
                total_epochs=training_config.epochs,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=amp_enabled,
                threshold=training_config.threshold,
            )
            validation_result = _run_epoch(
                model,
                validation_loader,
                validation_loss,
                device,
                epoch=epoch,
                total_epochs=training_config.epochs,
                use_amp=amp_enabled,
                threshold=training_config.threshold,
            )
            duration_seconds = time.perf_counter() - epoch_started
            learning_rate = float(optimizer.param_groups[0]["lr"])
            validation_f1 = float(validation_result["source"]["f1"])
            improved = validation_f1 > best_f1
            if improved:
                best_f1 = validation_f1
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            record = {
                "epoch": epoch,
                "duration_seconds": duration_seconds,
                "learning_rate": learning_rate,
                "improved": improved,
                "train_loss": train_result["loss"],
                "validation_loss": validation_result["loss"],
                "train_segment": train_result["segment"],
                "train_source": train_result["source"],
                "validation_segment": validation_result["segment"],
                "validation_source": validation_result["source"],
            }
            history.append(record)
            _log_epoch_tensorboard(
                writer,
                epoch,
                train_result,
                validation_result,
                learning_rate,
                duration_seconds,
                pos_weight,
                training_config.confusion_matrix_interval,
            )
            checkpoint_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "mel_config": asdict(mel_config),
                "training_config": asdict(training_config),
                "balance_classes": training_config.balance_classes,
                "pos_weight": pos_weight,
                "computed_pos_weight": computed_pos_weight,
                "rng_state": _rng_state(),
                "history": history,
                "class_mapping": {"real": 0, "fake": 1},
            }
            _atomic_torch_save(checkpoint_payload, last_path)
            if improved:
                _atomic_torch_save(checkpoint_payload, best_path)
            _save_history(history, output_dir)
            _atomic_json(
                {"best_epoch": best_epoch, "best_validation_source_f1": best_f1},
                output_dir / "best_epoch.json",
            )
            tqdm.write(
                f"Epoch {epoch + 1}/{training_config.epochs}: "
                f"{duration_seconds:.1f}s, train loss={record['train_loss']:.4f}, "
                f"val loss={record['validation_loss']:.4f}, val F1={validation_f1:.4f}, "
                f"val EER={validation_result['source']['eer']:.4f}, "
                f"лучшая эпоха={best_epoch + 1}; checkpoint сохранён"
            )
            if epochs_without_improvement >= training_config.early_stopping_patience:
                break
    finally:
        writer.close()
    return model, history


def evaluate_checkpoint(
    prepared_root: str | Path,
    checkpoint_path: str | Path,
    split: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict[str, Any]:
    """Оценить checkpoint; ``test_unseen`` вызывается только явно."""
    if split not in {"validation", "test_unseen"}:
        raise ValueError("Разрешена оценка только validation или test_unseen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mel_config = MelConfig(**checkpoint["mel_config"])
    manifest = pd.read_csv(Path(prepared_root) / "ruasd_segments_manifest.csv", low_memory=False)
    dataset = RuASDSegmentDataset(manifest, prepared_root, split, mel_config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    model = AudioDeepfakeCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = _run_epoch(
        model,
        loader,
        nn.BCEWithLogitsLoss(),
        device,
        epoch=0,
        total_epochs=1,
    )
    return {
        "split": split,
        "loss": result["loss"],
        "segment": result["segment"],
        "source": result["source"],
    }
