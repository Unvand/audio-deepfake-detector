"""Обучение одной CNN на log-Mel представлениях подготовленного RuASD.

Модуль не связан с GUI и не запускает обучение при импорте. ``test_unseen``
исключён из цикла обучения и validation; его оценка вызывается отдельно.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Нет сегментов для split={split}")
        if self.rows["segment_id"].duplicated().any():
            raise ValueError("segment_id должны быть уникальны")
        self.prepared_root = Path(prepared_root)
        self.mel_config = mel_config or MelConfig()
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
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
        log_mel = self.to_db(self.to_mel(tensor)).unsqueeze(0)
        # Нормализация выполняется отдельно для каждого примера и одинаково
        # для REAL/FAKE; исходные FLAC при этом не изменяются.
        log_mel = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-6)
        label = torch.tensor(float(str(row["label"]).lower() == "fake"))
        return log_mel, label, str(row["sample_id"])


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


def make_loaders(
    manifest: pd.DataFrame,
    prepared_root: str | Path,
    mel_config: MelConfig,
    training_config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Создать loaders только для train и validation."""
    train_dataset = RuASDSegmentDataset(manifest, prepared_root, "train", mel_config)
    validation_dataset = RuASDSegmentDataset(
        manifest, prepared_root, "validation", mel_config
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    common = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": training_config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels: list[float] = []
    scores: list[float] = []
    sample_ids: list[str] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for log_mel, target, batch_ids in tqdm(
            loader, desc="train" if training else "evaluate", leave=False
        ):
            log_mel = log_mel.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(log_mel)
            loss = loss_function(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            labels.extend(target.detach().cpu().numpy().tolist())
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            sample_ids.extend(batch_ids)
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
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    temporary = output_dir / "training_history.json.tmp"
    temporary.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "training_history.json")
    flattened = pd.json_normalize(history)
    csv_temporary = output_dir / "training_history.csv.tmp"
    flattened.to_csv(csv_temporary, index=False)
    csv_temporary.replace(output_dir / "training_history.csv")


def train_cnn(
    prepared_root: str | Path,
    output_dir: str | Path,
    mel_config: MelConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    resume: bool = True,
) -> tuple[AudioDeepfakeCNN, list[dict[str, Any]]]:
    """Обучить CNN, сохраняя best/last checkpoints после каждой эпохи."""
    prepared_root = Path(prepared_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(prepared_root / "ruasd_segments_manifest.csv")
    if set(manifest["split"].unique()) - {"train", "validation", "test_unseen"}:
        raise ValueError("Manifest содержит неизвестные split")
    mel_config = mel_config or MelConfig()
    training_config = training_config or TrainingConfig()
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
    loss_function = nn.BCEWithLogitsLoss()
    last_path = output_dir / "last_checkpoint.pth"
    best_path = output_dir / "best_model.pth"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_f1 = -1.0
    epochs_without_improvement = 0

    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["mel_config"] != asdict(mel_config):
            raise ValueError("MelConfig не совпадает с checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        history = checkpoint.get("history", [])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    for epoch in range(start_epoch, training_config.epochs):
        train_result = _run_epoch(
            model, train_loader, loss_function, device, optimizer, training_config.threshold
        )
        validation_result = _run_epoch(
            model, validation_loader, loss_function, device, None, training_config.threshold
        )
        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "train_segment": train_result["segment"],
            "train_source": train_result["source"],
            "validation_segment": validation_result["segment"],
            "validation_source": validation_result["source"],
        }
        history.append(record)
        validation_f1 = float(validation_result["source"]["f1"])
        improved = validation_f1 > best_f1
        if improved:
            best_f1 = validation_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "mel_config": asdict(mel_config),
            "training_config": asdict(training_config),
            "history": history,
            "class_mapping": {"real": 0, "fake": 1},
        }
        _atomic_torch_save(checkpoint_payload, last_path)
        if improved:
            _atomic_torch_save(checkpoint_payload, best_path)
        _save_history(history, output_dir)
        if epochs_without_improvement >= training_config.early_stopping_patience:
            break
    return model, history


def evaluate_checkpoint(
    prepared_root: str | Path,
    checkpoint_path: str | Path,
    split: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict[str, Any]:
    """Оценить сохранённую модель; ``test_unseen`` вызывается только явно."""
    if split not in {"validation", "test_unseen"}:
        raise ValueError("Разрешена оценка только validation или test_unseen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mel_config = MelConfig(**checkpoint["mel_config"])
    manifest = pd.read_csv(Path(prepared_root) / "ruasd_segments_manifest.csv")
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
    result = _run_epoch(model, loader, nn.BCEWithLogitsLoss(), device)
    return {
        "split": split,
        "loss": result["loss"],
        "segment": result["segment"],
        "source": result["source"],
    }
