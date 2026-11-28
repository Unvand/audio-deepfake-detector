"""Возобновляемый кэш frozen XLS-R и обучение головы без повторного энкодера."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from audio_deepfake_detector.src.training_metrics import (
    aggregate_source_scores,
    binary_metrics,
)
from audio_deepfake_detector.src.xlsr_training import (
    WaveformDataset,
    XLSRConfig,
    XLSRDetector,
)


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encoder_digest(encoder: nn.Module) -> str:
    """Привязать кэш к фактическим весам, а не только названию Hugging Face."""
    value = hashlib.sha256()
    for name, tensor in encoder.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def cache_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Выбрать только train/validation и проверить изоляцию источников."""
    rows = frame.loc[frame.split.isin(("train", "validation"))].copy()
    if rows.empty or set(rows.split) != {"train", "validation"}:
        raise ValueError("Нужны train и validation")
    if rows.segment_id.duplicated().any() or rows.audio_path.duplicated().any():
        raise ValueError("Повтор segment_id или audio_path")
    if rows.groupby("sample_id").split.nunique().gt(1).any() or rows.groupby("sample_id").label.nunique().gt(1).any():
        raise ValueError("Пересечение sample_id или противоречивые метки")
    if set(rows.label.str.lower()) != {"real", "fake"}:
        raise ValueError("Нужны REAL и FAKE")
    return rows.sort_values(["split", "segment_id"]).reset_index(drop=True)


def _identity(rows: pd.DataFrame, config: XLSRConfig, weights_sha: str) -> dict:
    fields = ["segment_id", "sample_id", "split", "label", "audio_path"]
    if "valid_duration_sec" in rows:
        fields.append("valid_duration_sec")
    content = rows[fields].to_json(orient="records", force_ascii=False).encode()
    return {"model_id": config.model_id, "encoder_sha256": weights_sha,
            "sample_rate": config.sample_rate, "seconds": config.seconds,
            "manifest_sha256": hashlib.sha256(content).hexdigest(),
            "segments": len(rows), "source_count": int(rows.sample_id.nunique()),
            "pooling": "masked_mean_v1", "dtype": "float32"}


def _verify_shard(path: Path, checksum: str, expected: pd.DataFrame) -> bool:
    if not path.is_file() or digest_file(path) != checksum:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (len(data["features"]) == len(expected)
                    and np.isfinite(data["features"]).all()
                    and data["segment_id"].tolist() == expected.segment_id.astype(str).tolist()
                    and data["sample_id"].tolist() == expected.sample_id.astype(str).tolist()
                    and data["label"].tolist() == expected.label.str.lower().eq("fake").astype(int).tolist())
    except (OSError, ValueError, KeyError):
        return False


def build_cache(frame: pd.DataFrame, root: Path, cache_dir: Path, model: XLSRDetector,
                config: XLSRConfig, *, shard_size: int = 256, local_work: Path | None = None) -> dict:
    """Один раз вычислить pooled features; готовые проверенные шарды пропустить."""
    if config.mode != "frozen" or shard_size < 1:
        raise ValueError("Кэш разрешён только для frozen и положительного shard_size")
    rows = cache_rows(frame)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError("Энкодер не заморожен")
    identity = _identity(rows, config, encoder_digest(model.encoder))
    metadata_path = cache_dir / "cache.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity or metadata["shard_size"] != shard_size:
            raise ValueError("Существующий кэш относится к другой модели, данным или параметрам")
    else:
        metadata = {"identity": identity, "shard_size": shard_size, "shards": {}}
    device = next(model.parameters()).device
    for split in ("train", "validation"):
        subset = rows.loc[rows.split.eq(split)].reset_index(drop=True)
        source = WaveformDataset(rows, root, split, replace(config, augmentation_probability=0))
        for start in tqdm(range(0, len(subset), shard_size), desc=f"XLS-R {split}", unit="шард"):
            batch = subset.iloc[start:start + shard_size]
            key = f"{split}/part-{start // shard_size:04d}.npz"
            target = cache_dir / key
            checksum = metadata["shards"].get(key)
            if checksum and _verify_shard(target, checksum, batch):
                continue
            loader = DataLoader(torch.utils.data.Subset(source, range(start, start + len(batch))),
                                batch_size=config.batch_size, shuffle=False)
            output = []
            with torch.inference_mode():
                for wave, mask, _, _ in loader:
                    wave, mask = wave.to(device), mask.to(device)
                    hidden = model.encoder(wave, attention_mask=mask.long()).last_hidden_state
                    out_mask = model.encoder._get_feature_vector_attention_mask(hidden.shape[1], mask.long())
                    output.append((hidden * out_mask[..., None]).sum(1).div(out_mask.sum(1).clamp_min(1)[:, None]).float().cpu().numpy())
            features = np.concatenate(output)
            if not np.isfinite(features).all():
                raise ValueError(f"Некорректные признаки {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            work_root = Path(local_work) if local_work else target.parent
            work_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=work_root, suffix=".npz", delete=False) as temporary:
                staging = Path(temporary.name)
            try:
                np.savez(staging, features=features, segment_id=np.asarray(batch.segment_id.astype(str).tolist(), dtype=str),
                         sample_id=np.asarray(batch.sample_id.astype(str).tolist(), dtype=str),
                         label=batch.label.str.lower().eq("fake").astype(np.int8).to_numpy())
                if local_work:
                    import shutil
                    shutil.copyfile(staging, target.with_suffix(".part"))
                    os.replace(target.with_suffix(".part"), target)
                else:
                    os.replace(staging, target)
                checksum = digest_file(target)
                if not _verify_shard(target, checksum, batch):
                    raise RuntimeError(f"Повреждён шард {key}")
                metadata["shards"][key] = checksum
                tmp = metadata_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(metadata, indent=2))
                os.replace(tmp, metadata_path)
            finally:
                staging.unlink(missing_ok=True)
    return metadata


def load_cache(cache_dir: Path, frame: pd.DataFrame, config: XLSRConfig, weights_sha: str,
               *, shard_size: int = 256) -> dict:
    """Полностью проверить кэш перед обучением; отсутствие шарда — ошибка."""
    rows = cache_rows(frame)
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / "cache.json").read_text())
    if metadata["identity"] != _identity(rows, config, weights_sha) or metadata["shard_size"] != shard_size:
        raise ValueError("Кэш несовместим")
    output = {}
    for split in ("train", "validation"):
        subset = rows.loc[rows.split.eq(split)].reset_index(drop=True)
        arrays, ids, labels = [], [], []
        for start in range(0, len(subset), shard_size):
            key = f"{split}/part-{start // shard_size:04d}.npz"
            path = cache_dir / key
            if not _verify_shard(path, metadata["shards"].get(key, ""), subset.iloc[start:start + shard_size]):
                raise ValueError(f"Отсутствует или повреждён шард {key}")
            with np.load(path, allow_pickle=False) as data:
                arrays.append(data["features"])
                ids.extend(data["sample_id"].tolist())
                labels.extend(data["label"].tolist())
        output[split] = (np.concatenate(arrays), np.asarray(labels), ids)
    return output


def verify_cached_logits(model: XLSRDetector, frame: pd.DataFrame, root: Path,
                         cached: dict, config: XLSRConfig, n: int = 4, atol: float = 1e-4) -> float:
    """Сверить logits прямого и кэшированного frozen пути на первых n строках."""
    model.eval()
    device = next(model.parameters()).device
    dataset = WaveformDataset(cache_rows(frame), root, "train", replace(config, augmentation_probability=0))
    diffs = []
    with torch.inference_mode():
        for index in range(min(n, len(dataset))):
            wave, mask, _, _ = dataset[index]
            direct = model(wave[None].to(device), mask[None].to(device))
            feature = torch.from_numpy(cached["train"][0][index:index + 1]).to(device)
            from_cache = model.head(feature).squeeze(-1)
            diffs.append(float((direct - from_cache).abs().max().cpu()))
    maximum = max(diffs)
    if maximum > atol:
        raise ValueError(f"Кэш и прямой XLS-R различаются: {maximum} > {atol}")
    return maximum


def train_cached_head(model: XLSRDetector, cached: dict, output: Path, config: XLSRConfig,
                      *, resume: bool = False, git_commit: str = "UNKNOWN", cache_identity: dict):
    """Обучить только голову и сохранить совместимый с notebook 11 полный checkpoint."""
    if config.mode != "frozen" or config.augmentation_probability:
        raise ValueError("Кэшированный запуск требует frozen без аугментаций")
    output = Path(output)
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = next(model.parameters()).device
    model.encoder.eval()
    head = model.head
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    pos_weight = float((cached["train"][1] == 0).sum() / (cached["train"][1] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    history, best, stale, start = [], -1., 0, 0
    last = output / "last_checkpoint.pth"
    if resume:
        state = torch.load(last, map_location="cpu", weights_only=False)
        old, new = dict(state["config"]), asdict(config)
        old.pop("epochs"); new.pop("epochs")
        if old != new or state["cache_identity"] != cache_identity or config.epochs < state["epoch"]:
            raise ValueError("Checkpoint несовместим с кэшем или конфигурацией")
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, best, stale, start = state["history"], state["best_f1"], state["stale"], state["epoch"]
    for epoch in range(start, config.epochs):
        started = time.monotonic()
        results = {}
        for split in ("train", "validation"):
            features, labels, ids = cached[split]
            head.train(split == "train")
            order = torch.randperm(len(labels)) if split == "train" else torch.arange(len(labels))
            losses, scores = [], np.empty(len(labels), dtype=float)
            for positions in tqdm(order.split(config.batch_size), desc=f"Epoch {epoch + 1} {split}"):
                x = torch.from_numpy(features[positions.numpy()]).to(device)
                y = torch.from_numpy(labels[positions.numpy()]).float().to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = head(x).squeeze(-1)
                    loss = criterion(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError("Loss NaN/Inf")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                losses.append(float(loss.detach().cpu()))
                scores[positions.numpy()] = torch.sigmoid(logits).detach().cpu().numpy()
            source_y, source_s = aggregate_source_scores(ids, labels, scores)
            results[split] = {"loss": float(np.mean(losses)), "segment": binary_metrics(labels, scores, config.threshold),
                              "source": binary_metrics(source_y, source_s, config.threshold)}
        f1 = results["validation"]["source"]["f1"]
        scheduler.step(f1)
        improved = f1 > best
        best, stale = (f1, 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch + 1, "train_loss": results["train"]["loss"],
                        "validation_loss": results["validation"]["loss"],
                        "validation_segment": results["validation"]["segment"],
                        "validation_source": results["validation"]["source"],
                        "seconds": time.monotonic() - started})
        state = {"head": head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": asdict(config), "cache_identity": cache_identity, "epoch": epoch + 1,
                 "best_f1": best, "stale": stale, "history": history, "git_commit": git_commit,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        temporary = last.with_suffix(".tmp")
        torch.save(state, temporary); temporary.replace(last)
        if improved:
            best_path = output / "best_model.pth"
            temporary = best_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch,
                        "best_f1": best, "git_commit": git_commit, "cache_identity": cache_identity}, temporary)
            temporary.replace(best_path)
        (output / "training_history.json").write_text(json.dumps(history, indent=2))
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        (output / "experiment_config.json").write_text(json.dumps(asdict(config), indent=2))
        print(f"Эпоха {epoch+1}: loss {results['train']['loss']:.4f}, validation F1 {f1:.4f}, EER {results['validation']['source']['eer']:.4f}")
        if stale >= config.patience:
            break
    return history
