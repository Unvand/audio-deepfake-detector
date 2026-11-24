"""Проверяемый локальный кэш подготовленных RuASD FLAC для Google Colab."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

MANIFEST_NAME = "ruasd_segments_manifest.csv"
INDEX_NAME = "cache_file_index.csv"
METADATA_NAME = "cache_metadata.json"
ALLOWED_SPLITS = {"train", "validation", "test_unseen"}
REQUIRED_COLUMNS = {
    "segment_id",
    "sample_id",
    "label",
    "split",
    "audio_path",
}


@dataclass(frozen=True)
class CacheStatus:
    """Результат полной проверки выбранной части кэша."""

    ready: bool
    selected_splits: tuple[str, ...]
    expected_files: int
    valid_files: int
    missing_files: int
    corrupted_files: int
    expected_bytes: int
    cached_bytes: int
    manifest_valid: bool
    message: str


@dataclass(frozen=True)
class CachePreparationResult:
    """Итог возобновляемого копирования."""

    ready: bool
    selected_splits: tuple[str, ...]
    expected_files: int
    copied_files: int
    skipped_files: int
    error_files: int
    copied_bytes: int
    errors: tuple[str, ...]


ProgressCallback = Callable[[int, int, dict[str, int]], None]


def _normalise_splits(splits: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(sorted(set(splits)))
    if not selected:
        raise ValueError("Выберите хотя бы один split")
    unknown = set(selected) - ALLOWED_SPLITS
    if unknown:
        raise ValueError(f"Неизвестные split: {sorted(unknown)}")
    return selected


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> pd.DataFrame:
    path = root / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Manifest не найден: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Manifest повреждён: {path}: {error}") from error
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"В manifest отсутствуют столбцы: {sorted(missing)}")
    if frame["segment_id"].duplicated().any():
        raise ValueError("Manifest содержит повторяющиеся segment_id")
    if set(frame["split"].unique()) - ALLOWED_SPLITS:
        raise ValueError("Manifest содержит неизвестные split")
    return frame


def _selected_rows(manifest: pd.DataFrame, splits: tuple[str, ...]) -> pd.DataFrame:
    selected = manifest[manifest["split"].isin(splits)].copy()
    if selected.empty:
        raise ValueError(f"Manifest не содержит выбранные split: {splits}")
    return selected


def _audio_is_valid(path: Path, expected_size: int, verify_decode: bool = True) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size:
        return False
    if not verify_decode:
        return True
    try:
        with sf.SoundFile(path) as stream:
            info = stream
            for _ in stream.blocks(blocksize=65_536, dtype="float32", always_2d=True):
                pass
    except (OSError, RuntimeError):
        return False
    return info.samplerate == 32_000 and info.frames > 0 and info.channels >= 1


def _load_index(cache_root: Path) -> pd.DataFrame:
    path = cache_root / INDEX_NAME
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["audio_path", "source_size", "sha256"])
    try:
        frame = pd.read_csv(path, dtype={"audio_path": str, "sha256": str})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["audio_path", "source_size", "sha256"])
    required = {"audio_path", "source_size", "sha256"}
    if not required.issubset(frame.columns) or frame["audio_path"].duplicated().any():
        return pd.DataFrame(columns=["audio_path", "source_size", "sha256"])
    sizes = pd.to_numeric(frame["source_size"], errors="coerce")
    frame = frame.loc[sizes.gt(0) & sizes.mod(1).eq(0)].copy()
    frame["source_size"] = sizes.loc[frame.index]
    return frame


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def cache_space_report(
    source_root: str | Path,
    cache_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    reserve_bytes: int = 2 * 1024**3,
) -> dict[str, int | bool]:
    """Оценить недостающие байты и проверить локальное свободное место."""
    source = Path(source_root)
    cache = Path(cache_root)
    selected_splits = _normalise_splits(splits)
    rows = _selected_rows(_read_manifest(source), selected_splits)
    index = _load_index(cache).set_index("audio_path", drop=False)
    required_bytes = 0
    total_bytes = 0
    for row in rows.itertuples(index=False):
        relative = str(row.audio_path)
        source_path = source / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"Исходный FLAC отсутствует: {source_path}")
        source_size = source_path.stat().st_size
        total_bytes += source_size
        indexed_size = (
            int(index.at[relative, "source_size"])
            if relative in index.index
            else source_size
        )
        if indexed_size != source_size or not _audio_is_valid(
            cache / relative, indexed_size, verify_decode=True
        ):
            required_bytes += source_size
    cache.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache).free
    return {
        "selected_files": len(rows),
        "dataset_bytes": total_bytes,
        "additional_bytes": required_bytes,
        "reserve_bytes": reserve_bytes,
        "free_bytes": free_bytes,
        "enough_space": free_bytes >= required_bytes + reserve_bytes,
    }


def check_local_cache(
    source_root: str | Path,
    cache_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    *,
    verify_decode: bool = True,
    verify_checksums: bool = False,
    compare_source_sizes: bool = True,
) -> CacheStatus:
    """Проверить manifest и только выбранные split частичного кэша."""
    source = Path(source_root)
    cache = Path(cache_root)
    selected_splits = _normalise_splits(splits)
    source_manifest_path = source / MANIFEST_NAME
    source_manifest = _read_manifest(source)
    rows = _selected_rows(source_manifest, selected_splits)
    manifest_valid = False
    try:
        cached_manifest = _read_manifest(cache)
        manifest_valid = (
            len(cached_manifest) == len(source_manifest)
            and _sha256(cache / MANIFEST_NAME) == _sha256(source_manifest_path)
        )
    except (FileNotFoundError, ValueError):
        cached_manifest = pd.DataFrame()

    index = _load_index(cache).set_index("audio_path", drop=False)
    valid = missing = corrupted = expected_bytes = cached_bytes = 0
    for row in rows.itertuples(index=False):
        relative = str(row.audio_path)
        destination = cache / relative
        if compare_source_sizes:
            source_path = source / relative
            if not source_path.is_file():
                corrupted += 1
                continue
            source_size = source_path.stat().st_size
            expected_bytes += source_size
        if relative not in index.index:
            if destination.exists():
                corrupted += 1
            else:
                missing += 1
            continue
        expected_size = int(index.at[relative, "source_size"])
        if compare_source_sizes and expected_size != source_size:
            corrupted += 1
            continue
        if not compare_source_sizes:
            expected_bytes += expected_size
        if not destination.exists():
            missing += 1
            continue
        if destination.is_file() and not destination.is_symlink():
            cached_bytes += destination.stat().st_size
        is_valid = _audio_is_valid(destination, expected_size, verify_decode)
        expected_hash = str(index.at[relative, "sha256"])
        if is_valid and verify_checksums:
            is_valid = (
                bool(expected_hash)
                and expected_hash != "nan"
                and _sha256(destination) == expected_hash
            )
        if is_valid:
            valid += 1
        else:
            corrupted += 1
    ready = manifest_valid and valid == len(rows)
    if ready:
        message = f"Кэш готов: {valid}/{len(rows)} файлов, split={selected_splits}"
    elif not cache.exists():
        message = "Локальный кэш отсутствует. Запустите ячейку подготовки кэша."
    else:
        message = (
            f"Кэш не готов: valid={valid}, missing={missing}, corrupted={corrupted}, "
            f"manifest_valid={manifest_valid}"
        )
    return CacheStatus(
        ready=ready,
        selected_splits=selected_splits,
        expected_files=len(rows),
        valid_files=valid,
        missing_files=missing,
        corrupted_files=corrupted,
        expected_bytes=expected_bytes,
        cached_bytes=cached_bytes,
        manifest_valid=manifest_valid,
        message=message,
    )


def prepare_local_cache(
    source_root: str | Path,
    cache_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    *,
    reserve_bytes: int = 2 * 1024**3,
    verify_checksums: bool = False,
    checkpoint_every: int = 250,
    progress_callback: ProgressCallback | None = None,
) -> CachePreparationResult:
    """Возобновляемо скопировать только отсутствующие/повреждённые FLAC."""
    source = Path(source_root)
    cache = Path(cache_root)
    selected_splits = _normalise_splits(splits)
    manifest = _read_manifest(source)
    rows = _selected_rows(manifest, selected_splits)
    space = cache_space_report(source, cache, selected_splits, reserve_bytes)
    if not space["enough_space"]:
        raise OSError(
            "Недостаточно места: нужно ещё "
            f"{space['additional_bytes']} байт + резерв {reserve_bytes}, "
            f"свободно {space['free_bytes']}"
        )

    cache.mkdir(parents=True, exist_ok=True)
    index = _load_index(cache)
    index_records = {
        str(row.audio_path): {
            "audio_path": str(row.audio_path),
            "source_size": int(row.source_size),
            "sha256": "" if pd.isna(row.sha256) else str(row.sha256),
        }
        for row in index.itertuples(index=False)
    }
    copied = skipped = errors_count = copied_bytes = 0
    errors: list[str] = []

    def save_index() -> None:
        ordered = pd.DataFrame(index_records.values())
        if not ordered.empty:
            ordered = ordered.sort_values("audio_path")
        _atomic_csv(ordered, cache / INDEX_NAME)

    progress = tqdm(total=len(rows), desc="Кэширование FLAC", unit="file")
    for processed, row in enumerate(rows.itertuples(index=False), start=1):
        relative = str(row.audio_path)
        source_path = source / relative
        destination = cache / relative
        try:
            if not source_path.is_file() or source_path.is_symlink():
                raise FileNotFoundError(f"Исходный файл отсутствует: {source_path}")
            source_size = source_path.stat().st_size
            known = index_records.get(relative)
            known_size = int(known["source_size"]) if known else source_size
            valid_existing = (
                known_size == source_size
                and _audio_is_valid(destination, known_size)
            )
            if valid_existing and verify_checksums:
                expected_hash = str(known.get("sha256", "")) if known else ""
                valid_existing = bool(expected_hash) and _sha256(destination) == expected_hash
            if valid_existing:
                skipped += 1
                if known is None:
                    index_records[relative] = {
                        "audio_path": relative,
                        "source_size": source_size,
                        "sha256": _sha256(destination) if verify_checksums else "",
                    }
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".part")
                shutil.copyfile(source_path, temporary, follow_symlinks=False)
                if not _audio_is_valid(temporary, source_size):
                    raise ValueError(f"Проверка FLAC после копирования не пройдена: {relative}")
                file_hash = _sha256(temporary) if verify_checksums else ""
                os.replace(temporary, destination)
                copied += 1
                copied_bytes += source_size
                index_records[relative] = {
                    "audio_path": relative,
                    "source_size": source_size,
                    "sha256": file_hash,
                }
        except (OSError, RuntimeError, ValueError) as error:
            errors_count += 1
            errors.append(f"{relative}: {type(error).__name__}: {error}")
            destination.with_name(destination.name + ".part").unlink(missing_ok=True)
        progress.update(1)
        progress.set_postfix(copied=copied, skipped=skipped, errors=errors_count)
        stats = {"copied": copied, "skipped": skipped, "errors": errors_count}
        if progress_callback is not None:
            progress_callback(processed, len(rows), stats)
        if processed % checkpoint_every == 0:
            save_index()
    progress.close()
    save_index()

    if errors_count == 0:
        temporary_manifest = cache / f"{MANIFEST_NAME}.tmp"
        shutil.copyfile(source / MANIFEST_NAME, temporary_manifest)
        os.replace(temporary_manifest, cache / MANIFEST_NAME)
    previous_splits: set[str] = set()
    if (cache / METADATA_NAME).exists():
        try:
            previous_splits = set(
                json.loads((cache / METADATA_NAME).read_text(encoding="utf-8")).get(
                    "cached_splits", []
                )
            )
        except (OSError, json.JSONDecodeError):
            previous_splits = set()
    metadata = {
        "manifest_sha256": _sha256(source / MANIFEST_NAME),
        "cached_splits": sorted(set(selected_splits) | previous_splits),
        "last_run": {
            "selected_splits": list(selected_splits),
            "copied": copied,
            "skipped": skipped,
            "errors": errors_count,
        },
    }
    _atomic_json(metadata, cache / METADATA_NAME)
    final_status = check_local_cache(
        source,
        cache,
        selected_splits,
        verify_decode=False,
        verify_checksums=verify_checksums,
    )
    return CachePreparationResult(
        ready=final_status.ready,
        selected_splits=selected_splits,
        expected_files=len(rows),
        copied_files=copied,
        skipped_files=skipped,
        error_files=errors_count,
        copied_bytes=copied_bytes,
        errors=tuple(errors),
    )


def require_ready_cache(
    source_root: str | Path,
    cache_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    *,
    compare_source_sizes: bool = True,
) -> Path:
    """Вернуть корень кэша либо запретить незаметный fallback на Google Drive."""
    status = check_local_cache(
        source_root, cache_root, splits,
        compare_source_sizes=compare_source_sizes,
        verify_checksums=not compare_source_sizes,
    )
    if not status.ready:
        raise RuntimeError(status.message + " Подготовьте/исправьте кэш перед обучением.")
    return Path(cache_root)


def status_as_dict(status: CacheStatus | CachePreparationResult) -> dict[str, Any]:
    """Подготовить dataclass для отображения в Colab."""
    return asdict(status)
