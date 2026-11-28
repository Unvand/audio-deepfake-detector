"""Второй эксперимент: waveform → XLS-R → masked pooling → REAL/FAKE."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from audio_deepfake_detector.src.ruasd_preparation import resample_mono
from audio_deepfake_detector.src.training_metrics import (
    aggregate_source_scores,
    binary_metrics,
)

MODEL_ID = "facebook/wav2vec2-xls-r-300m"


@dataclass(frozen=True)
class XLSRConfig:
    model_id: str = MODEL_ID
    sample_rate: int = 16_000
    seconds: float = 4.0
    dropout: float = 0.2
    mode: str = "frozen"
    unfreeze_last: int = 2
    batch_size: int = 4
    accumulation: int = 4
    epochs: int = 5
    head_lr: float = 1e-3
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42
    threshold: float = 0.5
    augmentation_probability: float = 0.0

    def validate(self) -> None:
        if self.sample_rate != 16_000 or self.seconds <= 0 or self.mode not in {"frozen", "partial"}:
            raise ValueError("XLS-R требует 16 кГц, положительное окно и frozen/partial")
        if min(self.batch_size, self.accumulation, self.epochs, self.patience) < 1:
            raise ValueError("Batch, accumulation, epochs и patience должны быть положительными")
        if self.unfreeze_last < 1:
            raise ValueError("Нужно разморозить хотя бы один последний слой")
        if not 0 <= self.augmentation_probability <= 1 or not 0 < self.threshold < 1:
            raise ValueError("Недопустимая вероятность аугментации или порог")


def audit_manifest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Проверить метки, пути и отсутствие пересечения исходных записей."""
    frame = pd.read_csv(path, low_memory=False, dtype={"sample_id": str, "segment_id": str})
    needed = {"sample_id", "segment_id", "audio_path", "split", "label"}
    if not needed <= set(frame) or frame[list(needed)].isna().any().any():
        raise ValueError("Manifest не содержит обязательных полей")
    if set(frame.label.str.lower()) != {"real", "fake"} or frame.segment_id.duplicated().any():
        raise ValueError("Недопустимые метки или повтор segment_id")
    if set(frame.split) != {"train", "validation", "test_unseen"}:
        raise ValueError("Ожидались train, validation, test_unseen")
    if frame.groupby("sample_id")["split"].nunique().gt(1).any() or frame.groupby("sample_id")["label"].nunique().gt(1).any():
        raise ValueError("Утечка sample_id или противоречивые метки")
    for name, split in zip(frame.audio_path, frame.split, strict=True):
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or p.parts[0] != split or p.suffix.lower() != ".flac":
            raise ValueError(f"Недопустимый путь: {name}")
    return frame, frame.groupby(["split", "label"]).agg(segments=("segment_id", "size"), sources=("sample_id", "nunique")).reset_index()


def select_pilot_sources(frame: pd.DataFrame, *, train_per_class: int = 500,
                         validation_per_class: int = 100, seed: int = 42) -> pd.DataFrame:
    """Выбрать целые записи до достижения числа сегментов каждого класса."""
    if min(train_per_class, validation_per_class) < 1:
        raise ValueError("Размер pilot должен быть положительным")
    if frame.segment_id.duplicated().any() or frame.groupby("sample_id").split.nunique().gt(1).any():
        raise ValueError("Дубли segment_id или пересечение sample_id")
    rng = np.random.default_rng(seed)
    selected = []
    for split, target in (("train", train_per_class), ("validation", validation_per_class)):
        for label in ("real", "fake"):
            group = frame.loc[frame.split.eq(split) & frame.label.str.lower().eq(label)]
            sizes = group.groupby("sample_id", sort=True).size()
            if sizes.sum() < target:
                raise ValueError(f"Недостаточно сегментов {split}/{label}")
            count = 0
            ids = []
            for index in rng.permutation(len(sizes)):
                ids.append(sizes.index[index])
                count += int(sizes.iloc[index])
                if count >= target:
                    break
            selected.append(group.loc[group.sample_id.isin(ids)])
    result = pd.concat(selected, ignore_index=True)
    if result.split.eq("test_unseen").any() or result.groupby("sample_id").split.nunique().gt(1).any():
        raise ValueError("Pilot содержит test_unseen или пересечение split")
    return result


class WaveformDataset(Dataset):
    """Читать подготовленные FLAC, не вычисляя Mel; mask исключает padding."""

    def __init__(self, frame: pd.DataFrame, root: Path, split: str, config: XLSRConfig):
        self.rows = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Нет данных {split}")
        self.root, self.split, self.config = Path(root), split, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        waveform, rate = sf.read(self.root / row.audio_path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if not np.isfinite(waveform).all():
            raise ValueError("Аудио содержит NaN/Inf")
        x = resample_mono(np.asarray(waveform, dtype=np.float32), rate, self.config.sample_rate)
        length = round(self.config.seconds * self.config.sample_rate)
        x = np.pad(x[:length], (0, max(0, length - len(x))))
        valid = float(row.get("valid_duration_sec", self.config.seconds))
        n_valid = min(length, max(1, round(valid * self.config.sample_rate)))
        mask = np.arange(length) < n_valid
        if self.split == "train" and np.random.random() < self.config.augmentation_probability:
            x = np.clip(x * np.random.uniform(.9, 1.1) + np.random.normal(0, .002, x.shape), -1, 1).astype(np.float32)
        label = float(str(row.label).lower() == "fake")
        return torch.from_numpy(x.copy()), torch.from_numpy(mask), torch.tensor(label), str(row.sample_id)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Усреднить только действительные выходные кадры энкодера."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class XLSRDetector(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, config: XLSRConfig):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(hidden_size, 1))
        self.config = config
        self.set_mode(config.mode)

    @classmethod
    def from_pretrained(cls, config: XLSRConfig):
        from transformers import Wav2Vec2Model

        encoder = Wav2Vec2Model.from_pretrained(config.model_id)
        if encoder.config.hidden_size <= 0:
            raise ValueError("Неизвестная размерность энкодера")
        return cls(encoder, encoder.config.hidden_size, config)

    def set_mode(self, mode: str) -> None:
        if mode not in {"frozen", "partial"}:
            raise ValueError("Неизвестный режим")
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if mode == "partial":
            layers = self.encoder.encoder.layers
            for layer in layers[-self.config.unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor):
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            hidden = self.encoder(waveform, attention_mask=attention_mask.long()).last_hidden_state
        if hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            out_mask = self.encoder._get_feature_vector_attention_mask(hidden.shape[1], attention_mask.long())
        else:
            out_mask = torch.nn.functional.interpolate(attention_mask[:, None].float(), size=hidden.shape[1], mode="nearest")[:, 0].bool()
        return self.head(masked_mean(hidden, out_mask)).squeeze(-1)


def run_epoch(model, loader, device, threshold, optimizer=None, accumulation=1, pos_weight=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training and not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()
    if training:
        optimizer.zero_grad(set_to_none=True)
    scores, labels, ids, losses = [], [], [], []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, (wave, mask, target, names) in enumerate(tqdm(loader, desc="train" if training else "validation")):
        wave, mask, target = wave.to(device), mask.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(wave, mask)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise ValueError("Loss содержит NaN/Inf")
            if training:
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
                if scaler is not None:
                    scaler.scale(loss / group_size).backward()
                else:
                    (loss / group_size).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(target.detach().cpu().int().tolist())
        ids.extend(names)
    y, s = np.asarray(labels), np.asarray(scores)
    source_y, source_s = aggregate_source_scores(ids, y, s)
    return {"loss": float(np.mean(losses)), "segment": binary_metrics(y, s, threshold),
            "source": binary_metrics(source_y, source_s, threshold), "scores": s, "labels": y, "sample_ids": ids,
            "n_segments": len(y), "n_sources": len(source_y), "duration_seconds": time.monotonic() - started,
            "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}


def train(root: Path, manifest: pd.DataFrame, output: Path, config: XLSRConfig, *, resume=False, git_commit="UNKNOWN"):
    """Обучить только train; validation выбирает лучшую модель по source F1."""
    config.validate()
    output = Path(output)
    if output.exists() and not resume and any(p.name != "pilot_selection.csv" for p in output.iterdir()):
        raise FileExistsError("Эксперимент уже существует; выберите другой run_id или resume")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    model = XLSRDetector.from_pretrained(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    groups = [{"params": model.head.parameters(), "lr": config.head_lr}]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": config.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    loaders = {split: DataLoader(WaveformDataset(manifest, root, split, config), batch_size=config.batch_size,
                                 shuffle=split == "train", num_workers=0) for split in ("train", "validation")}
    count = manifest.loc[manifest.split.eq("train"), "label"].str.lower().value_counts()
    pos_weight = torch.tensor(float(count["real"] / count["fake"]), device=device)
    start, best, stale, history = 0, -1., 0, []
    last = output / "last_checkpoint.pth"
    if resume:
        if not last.exists():
            raise FileNotFoundError("Checkpoint для resume отсутствует")
        state = torch.load(last, map_location="cpu", weights_only=False)
        old = dict(state["config"]); old.pop("epochs")
        new = asdict(config); new.pop("epochs")
        if old != new or config.epochs < state["epoch"] + 1:
            raise ValueError("Конфигурация несовместима с checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, best, stale, history = state["epoch"] + 1, state["best_f1"], state["stale"], state["history"]
    for epoch in range(start, config.epochs):
        train_result = run_epoch(model, loaders["train"], device, config.threshold, optimizer, config.accumulation, pos_weight, scaler)
        valid = run_epoch(model, loaders["validation"], device, config.threshold)
        scheduler.step(valid["source"]["f1"])
        history.append({"epoch": epoch + 1, "train_loss": train_result["loss"], "validation_loss": valid["loss"],
                        "validation_segment": valid["segment"], "validation_source": valid["source"],
                        "train_segments": train_result["n_segments"], "validation_segments": valid["n_segments"],
                        "train_sources": train_result["n_sources"], "validation_sources": valid["n_sources"],
                        "epoch_seconds": train_result["duration_seconds"] + valid["duration_seconds"],
                        "peak_cuda_mib": max(train_result["peak_cuda_mib"] or 0, valid["peak_cuda_mib"] or 0)})
        improved = valid["source"]["f1"] > best
        if improved:
            best, stale = valid["source"]["f1"], 0
        else:
            stale += 1
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict() if scaler is not None else None,
                 "config": asdict(config), "epoch": epoch, "best_f1": best, "stale": stale, "history": history,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "git_commit": git_commit, "torch_version": torch.__version__}
        torch.save(state, output / "last_checkpoint.pth.tmp")
        (output / "last_checkpoint.pth.tmp").replace(last)
        if improved:
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch,
                        "best_f1": best, "git_commit": git_commit}, output / "best_model.pth.tmp")
            (output / "best_model.pth.tmp").replace(output / "best_model.pth")
        (output / "training_history.json").write_text(json.dumps(history, indent=2))
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        (output / "experiment_config.json").write_text(json.dumps(asdict(config), indent=2))
        print(f"Epoch {epoch+1}: train loss={train_result['loss']:.4f}, validation loss={valid['loss']:.4f}, "
              f"Acc={valid['source']['accuracy']:.4f}, P={valid['source']['precision']:.4f}, "
              f"R={valid['source']['recall']:.4f}, F1={valid['source']['f1']:.4f}, "
              f"EER={valid['source']['eer']:.4f}, segments={valid['n_segments']}, "
              f"sources={valid['n_sources']}, seconds={history[-1]['epoch_seconds']:.1f}, "
              f"GPU MiB={history[-1]['peak_cuda_mib']:.1f}")
        if stale >= config.patience:
            break
    return history
