"""Поэтапная упаковка и восстановление готовых FLAC из TAR."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from audio_deepfake_detector.src.colab_cache import (
    INDEX_NAME,
    MANIFEST_NAME,
    _atomic_csv,
    _atomic_json,
    _audio_is_valid,
    _load_index,
    _normalise_splits,
    _read_manifest,
    _sha256,
)

GIB = 1024**3
DEFAULT_SHARD_BYTES = 3 * GIB // 2


def _paths(source, splits):
    manifest = _read_manifest(source)
    result = {}
    seen = set()
    selected = _normalise_splits(splits)
    if "test_unseen" in selected and len(selected) > 1:
        raise ValueError("Обрабатывайте test_unseen отдельно от train и validation")
    for split in selected:
        paths = manifest.loc[manifest["split"] == split, "audio_path"].tolist()
        if not paths:
            raise ValueError(f"Пустой split {split}")
        for name in paths:
            path = PurePosixPath(str(name))
            if (
                len(path.parts) < 2 or path.parts[0] != split or ".." in path.parts
                or path.as_posix() != name or path.suffix.lower() != ".flac"
                or "\\" in name or name in seen
            ):
                raise ValueError(f"Недопустимый или повторяющийся audio_path: {name}")
            seen.add(name)
        result[split] = paths
    return result


def _file(root, name):
    path = root / name
    if path.resolve() != path:
        raise ValueError(f"Символические ссылки запрещены: {path}")
    return path


def _directory_space(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _disjoint(*roots):
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"Используйте отдельные каталоги: {left}, {right}")


def _header(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    return info


def _packed_size(size):
    return ((size + 1024 + 10239) // 10240) * 10240


def _make_plan(source, paths, max_bytes):
    shards = []
    members = {}
    used = 0
    for name in tqdm(paths, desc="Размеры FLAC", unit="файл"):
        size = _file(source, name).stat().st_size
        cost = len(_header(name, size).tobuf(format=tarfile.PAX_FORMAT))
        cost += ((size + 511) // 512) * 512
        if members and _packed_size(used + cost) > max_bytes:
            shards.append({"members": members, "bytes": _packed_size(used)})
            members, used = {}, 0
        members[name] = size
        used += cost
    shards.append({"members": members, "bytes": _packed_size(used)})
    return shards


def _read_plan(directory, paths):
    plan = directory / "plan.json"
    if not plan.is_file():
        raise FileNotFoundError(f"План TAR отсутствует для {directory.name}. Сначала создайте архивы.")
    shards = json.loads(plan.read_text(encoding="utf-8"))["shards"]
    if [name for shard in shards for name in shard["members"]] != paths:
        raise ValueError(f"План TAR не соответствует manifest: {directory}")
    return shards


def _archive_paths(directory, number):
    stem = f"part-{number:04d}"
    return _file(directory, stem + ".tar"), _file(directory, stem + ".json")


def _receipt(archive, receipt_path, shard):
    if not receipt_path.is_file() or not archive.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(receipt, dict)
        or not isinstance(receipt.get("members"), dict)
        or archive.stat().st_size != shard["bytes"]
        or set(receipt["members"]) != set(shard["members"])
        or any(
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in [receipt.get("sha256"), *receipt["members"].values()]
        )
    ):
        return None
    return receipt


def _inspect_part(archive, receipt_path, shard, verify_sha, verify_contents):
    partial = archive.with_suffix(".tar.part")
    row = {
        "archive": archive.name, "split": archive.parent.name,
        "planned_bytes": shard["bytes"], "flac_count": len(shard["members"]),
        "size_bytes": None, "sha256": None, "partial_bytes": 0,
        "status": "Отсутствует", "error": None,
    }
    try:
        if partial.is_file():
            row["partial_bytes"] = partial.stat().st_size
        if not archive.is_file():
            row["status"] = "Незавершён" if partial.is_file() else "Отсутствует"
            return row
        row["size_bytes"] = archive.stat().st_size
        receipt = _receipt(archive, receipt_path, shard)
        if receipt is None:
            raise ValueError("Размер или JSON с контрольными суммами не совпадает с планом")
        row["sha256"] = receipt["sha256"]
        if verify_sha and _sha256(archive) != receipt["sha256"]:
            raise ValueError("SHA-256 архива не совпадает")
        if verify_contents:
            _unpack(archive, shard["members"], receipt["members"], {})
        row["status"] = "Готов" if verify_sha else "SHA-256 не проверена"
    except (OSError, ValueError, tarfile.TarError) as error:
        row.update(status="Повреждён", error=str(error))
    return row


def _build_inputs(source, archive_root, splits, max_bytes):
    version = archive_root / _sha256(source / MANIFEST_NAME)
    for split, paths in _paths(source, splits).items():
        directory = version / split
        if (directory / "plan.json").exists():
            shards = _read_plan(directory, paths)
        else:
            shards = _make_plan(source, paths, max_bytes)
        yield directory, shards


def build_tar_archives(
    source_root, archive_root, staging_root, splits=("train", "validation"), *,
    max_bytes=DEFAULT_SHARD_BYTES, archive_free_bytes=None, reserve_bytes=GIB,
    plan_only=False, verify_sha=False, verify_contents=False,
):
    """Оценить место или упаковать split с журналом завершённых частей на Drive."""
    source, destination, staging = (
        Path(root).resolve() for root in (source_root, archive_root, staging_root)
    )
    _disjoint(source, destination, staging)
    if max_bytes < 10240 or reserve_bytes < 0:
        raise ValueError("Некорректный размер TAR или резерв")
    plans = list(_build_inputs(source, destination, splits, max_bytes))
    pending = []
    rows = []
    for directory, shards in plans:
        parts = tqdm(shards, desc=f"Проверка {directory.name}", unit="TAR")
        for number, shard in enumerate(parts):
            archive, receipt_path = _archive_paths(directory, number)
            row = _inspect_part(
                archive, receipt_path, shard, verify_sha or not plan_only, verify_contents,
            )
            rows.append(row)
            if row["status"] not in {"Готов", "SHA-256 не проверена"}:
                pending.append((archive, receipt_path, shard))
    needed = sum(shard["bytes"] for _, _, shard in pending)
    peak = max((shard["bytes"] for _, _, shard in pending), default=0)
    free = _directory_space(destination) if archive_free_bytes is None else archive_free_bytes
    local_free = _directory_space(staging)
    report = {
        "pending_archives": len(pending),
        "planned_archives": len(rows),
        "ready_archives": len(rows) - len(pending),
        "verified_archives": sum(row["status"] == "Готов" for row in rows),
        "planned_bytes": sum(row["planned_bytes"] for row in rows),
        "partial_bytes_on_drive": sum(row["partial_bytes"] for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
        "archives": rows,
        "additional_drive_bytes": needed,
        "drive_free_bytes": free,
        "peak_local_tar_bytes": peak,
        "local_free_bytes": local_free,
        "reserve_bytes": reserve_bytes,
        "enough_space": not pending or (
            free >= needed + reserve_bytes and local_free >= peak + reserve_bytes
        ),
    }
    if plan_only:
        return report
    tqdm.write(
        f"TAR запланировано {len(rows)}, готово {report['ready_archives']}, "
        f"осталось {len(pending)}, ошибок проверки {report['errors']}"
    )
    if not report["enough_space"]:
        raise OSError(f"Недостаточно места для полного плана TAR: {report}")
    if not pending:
        return report
    for directory, shards in plans:
        _atomic_json({"shards": shards}, directory / "plan.json")
    staging.mkdir(parents=True, exist_ok=True)
    for archive, receipt_path, shard in tqdm(pending, desc="Упаковка", unit="TAR"):
        with tempfile.TemporaryDirectory(dir=staging) as work:
            local = Path(work) / archive.name
            hashes = {}
            with tarfile.open(local, "w", format=tarfile.PAX_FORMAT) as packed:
                for name, size in tqdm(
                    shard["members"].items(), desc=f"{archive.parent.name}/{archive.name}",
                    unit="FLAC", leave=False,
                ):
                    path = _file(source, name)
                    payload = path.read_bytes()
                    if len(payload) != size:
                        raise ValueError(f"Размер исходного FLAC изменился: {name}")
                    with sf.SoundFile(io.BytesIO(payload)) as audio:
                        if (
                            audio.format != "FLAC" or audio.samplerate != 32_000
                            or audio.frames <= 0 or len(audio.read()) != audio.frames
                        ):
                            raise ValueError(f"Не удалось проверить исходный FLAC: {name}")
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                    packed.addfile(_header(name, size), io.BytesIO(payload))
            if local.stat().st_size != shard["bytes"]:
                raise ValueError(f"Размер TAR не совпал с планом: {archive}")
            _unpack(local, shard["members"], hashes, {})
            checksum = _sha256(local)
            partial = _file(archive.parent, archive.name + ".part")
            shutil.copyfile(local, partial)
            if _sha256(partial) != checksum:
                raise ValueError(f"Контрольная сумма TAR на Drive не совпала: {archive}")
            partial.replace(archive)
            _atomic_json({"sha256": checksum, "members": hashes}, receipt_path)
            tqdm.write(f"Проверен и записан {archive.parent.name}/{archive.name}")
    return report


def _unpack(archive, sizes, hashes, destinations):
    """Проверить структуру и содержимое TAR, записать только запрошенные файлы."""
    with tarfile.open(archive, "r:") as packed:
        for name, size in sizes.items():
            member = packed.next()
            if member is None or not member.isfile() or (member.name, member.size) != (name, size):
                raise ValueError(f"Неожиданный элемент TAR, ожидался {name}")
            stream = packed.extractfile(member)
            if stream is None:
                raise ValueError(f"Не удалось прочитать файл внутри TAR: {name}")
            with stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != hashes[name]:
                raise ValueError(f"Повреждён файл внутри TAR: {name}")
            if name in destinations:
                target = destinations[name]
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = _file(target.parent, target.name + ".part")
                partial.write_bytes(payload)
                partial.replace(target)
        if packed.next() is not None:
            raise ValueError(f"Лишние элементы в {archive}")


def _published(source, archive_root, splits):
    version = archive_root / _sha256(source / MANIFEST_NAME)
    for split, paths in _paths(source, splits).items():
        directory = version / split
        for number, shard in enumerate(_read_plan(directory, paths)):
            archive, receipt_path = _archive_paths(directory, number)
            receipt = _receipt(archive, receipt_path, shard)
            if receipt is None:
                raise ValueError(f"TAR отсутствует или не подтверждён: {archive}")
            yield archive, shard, receipt


def list_ready_tar_archives(source_root, archive_root, splits=("train", "validation")):
    """Проверить план, наличие, размеры и JSON готовых частей без чтения TAR."""
    source = Path(source_root).resolve()
    archives = Path(archive_root).resolve()
    return tuple(archive for archive, _, _ in _published(source, archives, splits))


def restore_tar_cache(
    source_root, archive_root, cache_root, splits=("train", "validation"), *,
    reserve_bytes=GIB, progress_callback=None,
):
    """Восстановить только недостающие файлы из проверенных частей TAR."""
    source, archive_root, cache = (
        Path(root).resolve() for root in (source_root, archive_root, cache_root)
    )
    _disjoint(source, archive_root, cache)
    archives = list(_published(source, archive_root, splits))
    records = {
        row["audio_path"]: row for row in _load_index(cache).to_dict("records")
    }
    jobs = []
    total = copied = skipped = transferred = 0
    for archive, shard, receipt in tqdm(archives, desc="Проверка локальных FLAC"):
        missing = {}
        for name, size in shard["members"].items():
            target = _file(cache, name)
            expected = receipt["members"][name]
            if not _audio_is_valid(target, size, verify_decode=False) or _sha256(target) != expected:
                missing[name] = target
        jobs.append((archive, shard, receipt, missing))
        total += len(shard["members"])
    needed = sum(shard["members"][name] for _, shard, _, missing in jobs for name in missing)
    peak = max((shard["bytes"] for _, shard, _, missing in jobs if missing), default=0)
    if needed and _directory_space(cache) < needed + peak + reserve_bytes:
        raise OSError(f"Недостаточно места в /content для FLAC ({needed} Б), TAR ({peak} Б) и резерва")
    cache.mkdir(parents=True, exist_ok=True)
    for archive, shard, receipt, missing in tqdm(jobs, desc="Восстановление", unit="TAR"):
        if missing:
            with tempfile.TemporaryDirectory(dir=cache) as work:
                local = Path(work) / archive.name
                shutil.copyfile(archive, local)
                if _sha256(local) != receipt["sha256"]:
                    raise ValueError(f"Повреждён TAR: {archive}")
                _unpack(local, shard["members"], receipt["members"], missing)
                transferred += shard["bytes"]
        for name, size in shard["members"].items():
            records[name] = {
                "audio_path": name, "source_size": size, "sha256": receipt["members"][name],
            }
        _atomic_csv(pd.DataFrame(records.values()), cache / INDEX_NAME)
        copied += len(missing)
        skipped += len(shard["members"]) - len(missing)
        if progress_callback is not None:
            progress_callback(copied + skipped, total, {"copied": copied, "skipped": skipped, "errors": 0})
    manifest = cache / (MANIFEST_NAME + ".tmp")
    shutil.copyfile(source / MANIFEST_NAME, manifest)
    manifest.replace(cache / MANIFEST_NAME)
    return {"restored_files": copied, "skipped_files": skipped, "transferred_tar_bytes": transferred}


def benchmark_tar_transfer(source_root, archive_root, staging_root, split="train"):
    """Замерить один готовый TAR и те же FLAC в отдельном временном каталоге."""
    source, archive_root, staging = (
        Path(root).resolve() for root in (source_root, archive_root, staging_root)
    )
    _disjoint(source, archive_root, staging)
    archive, shard, receipt = min(
        _published(source, archive_root, (split,)), key=lambda part: part[1]["bytes"]
    )
    if _directory_space(staging) < shard["bytes"] + sum(shard["members"].values()) + GIB:
        raise OSError("Недостаточно места для benchmark")
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as work:
        root = Path(work)
        outputs = {name: root / name for name in shard["members"]}
        start = time.perf_counter()
        for name, target in tqdm(outputs.items(), desc="Benchmark FLAC"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_file(source, name), target)
        files_seconds = time.perf_counter() - start
    with tempfile.TemporaryDirectory(dir=staging) as work:
        local = Path(work) / archive.name
        start = time.perf_counter()
        shutil.copyfile(archive, local)
        tar_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if _sha256(local) != receipt["sha256"]:
            raise ValueError(f"Повреждён TAR: {archive}")
        outputs = {name: Path(work) / name for name in shard["members"]}
        _unpack(local, shard["members"], receipt["members"], outputs)
        validation_seconds = time.perf_counter() - start
    return {
        "files": len(shard["members"]),
        "flac_copy_seconds": files_seconds,
        "tar_copy_seconds": tar_seconds,
        "tar_verify_and_extract_seconds": validation_seconds,
        "tar_total_seconds": tar_seconds + validation_seconds,
    }
