"""Потоковая подготовка RuASD по зафиксированному CSV-манифесту.

Модуль не запускает загрузку при импорте. Массовая обработка начинается только
после явного вызова :func:`prepare_ruasd_dataset` из Colab notebook.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import ceil, gcd
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm.auto import tqdm

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "label",
    "split",
    "subset",
    "model",
    "duration",
    "shard_id",
    "__key__",
    "__url__",
}
ALLOWED_SPLITS = {"train", "validation", "test_unseen"}
ALLOWED_LABELS = {"real", "fake"}


@dataclass(frozen=True)
class PreparationConfig:
    """Параметры воспроизводимой подготовки аудио."""

    target_sample_rate: int = 32_000
    segment_seconds: float = 4.0
    output_format: str = "FLAC"
    flac_subtype: str = "PCM_16"
    dataset_revision: str = "fcbc87c57b54ef4f58e1135e2813f6f000c2b739"
    num_shards: int = 250

    def validate(self) -> None:
        """Проверить значения до начала длительной обработки."""
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate должен быть положительным")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds должен быть положительным")
        if self.output_format.upper() not in {"FLAC", "WAV"}:
            raise ValueError("Поддерживаются только FLAC и WAV")
        if self.num_shards <= 0:
            raise ValueError("num_shards должен быть положительным")


@dataclass(frozen=True)
class DecodedAudio:
    """Декодированный mono-сигнал вместе с исходной частотой."""

    waveform: np.ndarray
    sample_rate: int


def validate_ruasd_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    """Проверить схему, уникальность ID и неизменность готовых split."""
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"В манифесте отсутствуют поля: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Манифест пуст")
    if manifest["sample_id"].isna().any():
        raise ValueError("sample_id не должен быть пустым")
    duplicate_count = int(manifest["sample_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Найдены повторяющиеся sample_id: {duplicate_count}")

    splits = set(manifest["split"].astype(str).unique())
    labels = set(manifest["label"].astype(str).str.lower().unique())
    if not splits.issubset(ALLOWED_SPLITS):
        raise ValueError(f"Неизвестные split: {sorted(splits - ALLOWED_SPLITS)}")
    if not labels.issubset(ALLOWED_LABELS):
        raise ValueError(f"Неизвестные label: {sorted(labels - ALLOWED_LABELS)}")

    split_label = pd.crosstab(manifest["split"], manifest["label"])
    split_ids = {
        split: set(group["sample_id"].astype(str))
        for split, group in manifest.groupby("split")
    }
    overlap = set()
    split_names = sorted(split_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap.update(split_ids[left].intersection(split_ids[right]))
    if overlap:
        raise ValueError("sample_id пересекаются между split")

    return {
        "records": len(manifest),
        "unknown_duration": int((manifest["duration"] <= 0).sum()),
        "unique_sample_ids": int(manifest["sample_id"].nunique()),
        "split_label": split_label.to_dict(),
    }


def select_preview_examples(
    manifest: pd.DataFrame,
    include_unseen: bool = False,
) -> pd.DataFrame:
    """Выбрать по одному train-примеру нужных REAL-источников и FAKE-моделей."""
    real_sources = ["CommonVoice", "GOLOS", "M-AILABS", "SOVA-AB"]
    train_fake = [
        "11labs",
        "bark_small_tts",
        "fishsTTS",
        "glados2",
        "grad_tts",
        "pyx3TTS",
        "sf5TTS",
        "tts_ru_free_hf_vits_high_multispeaker",
    ]
    requested = real_sources + train_fake
    if include_unseen:
        requested += ["ruxTTS", "xtts_ru_ipa"]

    rows: list[pd.Series] = []
    for subset in requested:
        allowed_split = "test_unseen" if subset in {"ruxTTS", "xtts_ru_ipa"} else "train"
        candidates = manifest[
            (manifest["subset"] == subset) & (manifest["split"] == allowed_split)
        ]
        if candidates.empty:
            raise ValueError(f"Не найден пример для subset={subset}, split={allowed_split}")
        rows.append(candidates.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def select_test_records(manifest: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Детерминированно выбрать небольшой REAL/FAKE-набор только из train."""
    if limit < 2:
        raise ValueError("Для TEST MODE требуется как минимум две записи")
    train = manifest[manifest["split"] == "train"]
    real_count = limit // 2
    fake_count = limit - real_count
    selected = pd.concat(
        [
            train[train["label"].str.lower() == "real"].head(real_count),
            train[train["label"].str.lower() == "fake"].head(fake_count),
        ],
        ignore_index=True,
    )
    if len(selected) != limit:
        raise ValueError("Недостаточно REAL/FAKE-записей для TEST MODE")
    return selected


def decoder_to_mono(audio_decoder: Any) -> DecodedAudio:
    """Получить numpy mono-сигнал из ``datasets`` AudioDecoder."""
    decoded = audio_decoder.get_all_samples()
    data = decoded.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    waveform = np.asarray(data, dtype=np.float32)
    if waveform.ndim == 2:
        # TorchCodec возвращает (channels, samples). Эвристика также допускает
        # массив (samples, channels) от тестовых/альтернативных декодеров.
        channel_axis = 0 if waveform.shape[0] <= 8 else 1
        waveform = waveform.mean(axis=channel_axis)
    elif waveform.ndim != 1:
        raise ValueError(f"Неожиданная форма аудио: {waveform.shape}")
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Декодированное аудио пусто или содержит NaN/Inf")
    return DecodedAudio(waveform=waveform.astype(np.float32), sample_rate=int(decoded.sample_rate))


def resample_mono(
    waveform: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = 32_000,
) -> np.ndarray:
    """Качественно ресемплировать mono-сигнал полифазным фильтром."""
    if waveform.ndim != 1:
        raise ValueError("Ожидается одномерный mono-сигнал")
    if original_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Частоты дискретизации должны быть положительными")
    if original_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=True)
    divisor = gcd(original_sample_rate, target_sample_rate)
    return resample_poly(
        waveform,
        up=target_sample_rate // divisor,
        down=original_sample_rate // divisor,
    ).astype(np.float32)


def segment_with_padding(
    waveform: np.ndarray,
    sample_rate: int = 32_000,
    segment_seconds: float = 4.0,
) -> list[tuple[np.ndarray, float]]:
    """Нарезать сигнал без перекрытия; вернуть сегменты и валидную длительность."""
    segment_samples = round(sample_rate * segment_seconds)
    if waveform.ndim != 1 or segment_samples <= 0:
        raise ValueError("Некорректный сигнал или длина сегмента")
    if waveform.size == 0:
        raise ValueError("Нельзя сегментировать пустое аудио")
    result: list[tuple[np.ndarray, float]] = []
    for start in range(0, waveform.size, segment_samples):
        part = waveform[start : start + segment_samples]
        valid_seconds = part.size / sample_rate
        if part.size < segment_samples:
            part = np.pad(part, (0, segment_samples - part.size))
        result.append((part.astype(np.float32), valid_seconds))
    return result


def _safe_stem(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:80]
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def _atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def estimate_prepared_size(
    manifest: pd.DataFrame,
    config: PreparationConfig,
    flac_ratio: float = 0.65,
) -> dict[str, int | float]:
    """Оценить число сегментов и размер; неизвестные длительности не выдумывать."""
    known = manifest[manifest["duration"] > 0]["duration"].astype(float)
    segments = int(sum(ceil(value / config.segment_seconds) for value in known))
    pcm_bytes = segments * round(config.segment_seconds * config.target_sample_rate) * 2
    return {
        "known_records": len(known),
        "unknown_records": int((manifest["duration"] <= 0).sum()),
        "known_duration_hours": float(known.sum() / 3600),
        "estimated_segments_known": segments,
        "estimated_pcm16_bytes_known": pcm_bytes,
        "rough_flac_bytes_known": int(pcm_bytes * flac_ratio),
    }


def disk_space_report(
    output_root: str | Path,
    estimated_bytes: int,
    reserve_bytes: int = 2 * 1024**3,
) -> dict[str, int | bool]:
    """Проверить место с резервом под checkpoints и результаты."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    return {
        "free_bytes": free,
        "estimated_bytes": int(estimated_bytes),
        "reserve_bytes": int(reserve_bytes),
        "enough_space": free >= estimated_bytes + reserve_bytes,
    }


class RuASDShardReader:
    """Читать каждый нужный логический shard один раз за проход."""

    def __init__(self, token: str, config: PreparationConfig) -> None:
        if not token:
            raise ValueError("HF_TOKEN не задан")
        self.token = token
        self.config = config

    def iter_decoded(
        self, records: pd.DataFrame
    ) -> Iterator[tuple[dict[str, Any], DecodedAudio | None, str | None]]:
        """Выдать строки manifest с аудио либо сообщением об ошибке."""
        from datasets import load_dataset

        dataset = load_dataset(
            "MTUCI/RuASD",
            split="train",
            streaming=True,
            token=self.token,
            revision=self.config.dataset_revision,
        )
        for shard_id, shard_rows in records.groupby("shard_id", sort=True):
            wanted = {
                str(row.sample_id): row._asdict()
                for row in shard_rows.itertuples(index=False)
            }
            shard_dataset = dataset.shard(
                num_shards=self.config.num_shards,
                index=int(shard_id),
            )
            shard_bar = tqdm(
                total=len(wanted),
                desc=f"Shard {int(shard_id):03d}",
                unit="file",
                leave=False,
            )
            for sample in shard_dataset:
                metadata = sample.get("json", {})
                sample_id = str(metadata.get("sample_id", sample.get("__key__", "")))
                row = wanted.pop(sample_id, None)
                if row is None:
                    continue
                try:
                    decoded = decoder_to_mono(sample["wav"])
                    yield row, decoded, None
                # Сторонний decoder может выбросить разные типы ошибок. Любая
                # ошибка одного файла журналируется и не останавливает shard.
                except Exception as error:  # noqa: BLE001
                    yield row, None, f"{type(error).__name__}: {error}"
                shard_bar.update(1)
                if not wanted:
                    break
            shard_bar.close()
            for row in wanted.values():
                yield row, None, "sample_id не найден при чтении shard"


def _completed_sample_ids(
    segments: pd.DataFrame, output_root: Path, config: PreparationConfig
) -> set[str]:
    if segments.empty:
        return set()
    complete: set[str] = set()
    expected_frames = round(config.target_sample_rate * config.segment_seconds)
    for sample_id, rows in segments.groupby("sample_id"):
        valid = True
        for relative_path in rows["audio_path"]:
            path = output_root / str(relative_path)
            try:
                info = sf.info(path)
                valid &= info.samplerate == config.target_sample_rate
                valid &= info.frames == expected_frames
            except (OSError, RuntimeError):
                valid = False
        if valid:
            complete.add(str(sample_id))
    return complete


def _write_audio_atomic(path: Path, waveform: np.ndarray, config: PreparationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    sf.write(
        temporary,
        waveform,
        config.target_sample_rate,
        format=config.output_format.upper(),
        subtype=config.flac_subtype if config.output_format.upper() == "FLAC" else "PCM_16",
    )
    temporary.replace(path)


def prepare_ruasd_dataset(
    manifest: pd.DataFrame,
    output_root: str | Path,
    token: str,
    config: PreparationConfig | None = None,
    *,
    full_mode: bool = False,
    test_limit: int = 8,
    checkpoint_every: int = 10,
    reader_factory: Callable[[str, PreparationConfig], Any] = RuASDShardReader,
) -> dict[str, Any]:
    """Подготовить RuASD с журналом, атомарной записью и возобновлением.

    ``full_mode=False`` принципиально ограничивает запуск небольшим TEST MODE.
    Перед FULL MODE notebook дополнительно требует явного подтверждения.
    """
    config = config or PreparationConfig()
    config.validate()
    validate_ruasd_manifest(manifest)
    selected = manifest.copy() if full_mode else select_test_records(manifest, test_limit)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    config_path = output / "preprocessing_config.json"
    config_payload = asdict(config)
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config_payload:
            raise ValueError(
                "Конфигурация отличается от существующей. Используйте новый output_root."
            )
    else:
        _atomic_json(config_payload, config_path)

    segments_path = output / "ruasd_segments_manifest.csv"
    progress_path = output / "processing_progress.csv"
    errors_path = output / "processing_errors.csv"
    segments = _load_existing_csv(segments_path)
    progress = _load_existing_csv(progress_path)
    errors = _load_existing_csv(errors_path)
    completed = _completed_sample_ids(segments, output, config)
    pending = selected[~selected["sample_id"].astype(str).isin(completed)].copy()

    segment_rows = segments.to_dict("records") if not segments.empty else []
    progress_rows = progress.to_dict("records") if not progress.empty else []
    error_rows = errors.to_dict("records") if not errors.empty else []
    timestamp = lambda: datetime.now(UTC).isoformat()

    def save_state() -> None:
        _atomic_dataframe_csv(pd.DataFrame(segment_rows), segments_path)
        _atomic_dataframe_csv(pd.DataFrame(progress_rows), progress_path)
        _atomic_dataframe_csv(pd.DataFrame(error_rows), errors_path)

    reader = reader_factory(token, config)
    decoded_items = reader.iter_decoded(pending) if not pending.empty else iter(())
    main_bar = tqdm(total=len(pending), desc="Исходные файлы", unit="file")
    for processed_count, (row, decoded, error) in enumerate(
        decoded_items, start=1
    ):
        sample_id = str(row["sample_id"])
        # Удаляем старые строки этого sample при повторе после ошибки/обрыва.
        segment_rows = [item for item in segment_rows if str(item["sample_id"]) != sample_id]
        if error or decoded is None:
            message = error or "Неизвестная ошибка декодирования"
            error_rows.append(
                {"sample_id": sample_id, "shard_id": row["shard_id"], "error": message, "time": timestamp()}
            )
            progress_rows.append(
                {"sample_id": sample_id, "status": "error", "segments": 0, "time": timestamp()}
            )
        else:
            created_paths: list[Path] = []
            try:
                resampled = resample_mono(
                    decoded.waveform, decoded.sample_rate, config.target_sample_rate
                )
                parts = segment_with_padding(
                    resampled, config.target_sample_rate, config.segment_seconds
                )
                stem = _safe_stem(sample_id)
                extension = ".flac" if config.output_format.upper() == "FLAC" else ".wav"
                new_rows: list[dict[str, Any]] = []
                for segment_index, (part, valid_seconds) in enumerate(parts):
                    segment_id = f"{stem}_{segment_index:04d}"
                    relative = Path(str(row["split"])) / f"{segment_id}{extension}"
                    destination = output / relative
                    _write_audio_atomic(destination, part, config)
                    created_paths.append(destination)
                    new_rows.append(
                        {
                            "segment_id": segment_id,
                            "sample_id": sample_id,
                            "label": str(row["label"]).lower(),
                            "split": row["split"],
                            "subset": row["subset"],
                            "model": row.get("model", ""),
                            "shard_id": int(row["shard_id"]),
                            "original_sample_rate": decoded.sample_rate,
                            "target_sample_rate": config.target_sample_rate,
                            "original_duration": decoded.waveform.size / decoded.sample_rate,
                            "segment_index": segment_index,
                            "start_sec": segment_index * config.segment_seconds,
                            "valid_duration_sec": valid_seconds,
                            "segment_duration_sec": config.segment_seconds,
                            "audio_path": relative.as_posix(),
                            "processing_status": "success",
                        }
                    )
                segment_rows.extend(new_rows)
                progress_rows.append(
                    {"sample_id": sample_id, "status": "success", "segments": len(parts), "time": timestamp()}
                )
            # Ошибки codec, диска и отдельных файлов должны попасть в журнал,
            # после чего обработка остальных записей продолжается.
            except Exception as processing_error:  # noqa: BLE001
                for created_path in created_paths:
                    created_path.unlink(missing_ok=True)
                message = f"{type(processing_error).__name__}: {processing_error}"
                error_rows.append(
                    {"sample_id": sample_id, "shard_id": row["shard_id"], "error": message, "time": timestamp()}
                )
                progress_rows.append(
                    {"sample_id": sample_id, "status": "error", "segments": 0, "time": timestamp()}
                )
        main_bar.update(1)
        main_bar.set_postfix(
            segments=len(segment_rows), errors=len(error_rows), remaining=len(pending) - processed_count
        )
        if processed_count % checkpoint_every == 0:
            save_state()
    main_bar.close()
    save_state()

    summary = {
        "mode": "FULL" if full_mode else "TEST",
        "selected_records": len(selected),
        "already_complete_before_run": int(len(selected) - len(pending)),
        "processed_this_run": len(pending),
        "successful_source_files": int(
            pd.DataFrame(progress_rows).query("status == 'success'")["sample_id"].nunique()
            if progress_rows
            else 0
        ),
        "segments": len(segment_rows),
        "errors_logged": len(error_rows),
        "updated_at": timestamp(),
    }
    _atomic_json(summary, output / "preprocessing_summary.json")
    return summary


def verify_prepared_dataset(
    output_root: str | Path,
    expected_sample_rate: int = 32_000,
    expected_seconds: float = 4.0,
) -> dict[str, Any]:
    """Проверить manifest, FLAC-файлы, длину и числовую корректность сигналов."""
    root = Path(output_root)
    manifest_path = root / "ruasd_segments_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    frame = pd.read_csv(manifest_path)
    required = {"segment_id", "sample_id", "label", "split", "audio_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"В manifest сегментов отсутствуют поля: {sorted(missing)}")
    if frame["segment_id"].duplicated().any():
        raise ValueError("Найдены повторяющиеся segment_id")
    expected_frames = round(expected_sample_rate * expected_seconds)
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        path = root / row.audio_path
        try:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            if sample_rate != expected_sample_rate or audio.shape[0] != expected_frames or not np.isfinite(audio).all():
                failures.append(str(row.segment_id))
        except (OSError, RuntimeError):
            failures.append(str(row.segment_id))
    return {
        "segments": len(frame),
        "source_files": int(frame["sample_id"].nunique()),
        "failures": failures,
        "valid": not failures,
    }
