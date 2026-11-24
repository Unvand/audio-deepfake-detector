"""Проверка TAR-кэша на синтетических FLAC."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from audio_deepfake_detector.src.colab_cache import MANIFEST_NAME, check_local_cache
from audio_deepfake_detector.src.colab_tar_cache import (
    benchmark_tar_transfer,
    build_tar_archives,
    list_ready_tar_archives,
    restore_tar_cache,
)


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source, archives, staging, cache = (
        tmp_path / name for name in ("source", "archives", "staging", "cache")
    )
    rows = []
    rng = np.random.default_rng(42)
    for split in ("train", "validation", "test_unseen"):
        (source / split).mkdir(parents=True)
        for index in range(2):
            name = f"{split}/{'x' * 110}-{index}.flac"
            sf.write(source / name, rng.uniform(-0.9, 0.9, 6_000), 32_000)
            rows.append({
                "segment_id": name, "sample_id": name, "label": "real",
                "split": split, "audio_path": name,
            })
    pd.DataFrame(rows).to_csv(source / MANIFEST_NAME, index=False)
    return source, archives, staging, cache


def test_space_plan_blocks_writes(roots) -> None:
    source, archives, staging, _ = roots
    report = build_tar_archives(
        source, archives, staging, archive_free_bytes=1, reserve_bytes=0, plan_only=True,
    )
    assert not report["enough_space"]
    assert report["additional_drive_bytes"] > 1
    with pytest.raises(OSError, match="Недостаточно места"):
        build_tar_archives(source, archives, staging, archive_free_bytes=1, reserve_bytes=0)
    assert not archives.exists()
    assert not staging.exists()


def test_shards_resume_and_restore_only_missing(roots) -> None:
    source, archives, staging, cache = roots
    options = {"max_bytes": 20480, "reserve_bytes": 0}
    build_tar_archives(source, archives, staging, **options)
    parts = sorted(archives.rglob("*.tar"))
    assert len(parts) == 4
    assert all(part.stat().st_size <= 20480 for part in parts)
    assert all(part.parent.name != "test_unseen" for part in parts)
    assert sorted(list_ready_tar_archives(source, archives)) == parts
    original_times = [part.stat().st_mtime_ns for part in parts]
    parts[0].unlink()
    report = build_tar_archives(source, archives, staging, **options)
    assert report["pending_archives"] == 1
    assert [part.stat().st_mtime_ns for part in parts[1:]] == original_times[1:]

    first = restore_tar_cache(source, archives, cache, reserve_bytes=0)
    assert first["restored_files"] == 4
    assert not (cache / "test_unseen").exists()
    assert check_local_cache(source, cache, verify_checksums=True).ready
    assert check_local_cache(
        source, cache, verify_checksums=True, compare_source_sizes=False,
    ).ready
    assert (cache / MANIFEST_NAME).read_bytes() == (source / MANIFEST_NAME).read_bytes()
    result = restore_tar_cache(source, archives, cache, reserve_bytes=0)
    assert result["transferred_tar_bytes"] == 0
    train_file = next((cache / "train").glob("*.flac"))
    train_file.write_bytes(b"x" * train_file.stat().st_size)
    repaired = restore_tar_cache(source, archives, cache, reserve_bytes=0)
    assert repaired["restored_files"] == 1
    assert repaired["transferred_tar_bytes"] == 20480
    assert train_file.read_bytes() == (source / train_file.relative_to(cache)).read_bytes()


def test_rejects_corrupt_tar_and_benchmarks_repaired_archive(roots) -> None:
    source, archives, staging, cache = roots
    build_tar_archives(source, archives, staging, ("train",), reserve_bytes=0)
    archive = next(archives.rglob("*.tar"))
    with archive.open("r+b") as stream:
        stream.seek(2000)
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="Повреждён TAR"):
        restore_tar_cache(source, archives, cache, ("train",), reserve_bytes=0)
    assert not list(cache.rglob("*.flac"))
    assert build_tar_archives(
        source, archives, staging, ("train",), reserve_bytes=0,
    )["pending_archives"] == 1
    result = benchmark_tar_transfer(source, archives, staging)
    assert result["files"] == 2
    assert result["tar_total_seconds"] >= result["tar_copy_seconds"] > 0
    assert result["flac_copy_seconds"] > 0


def test_test_unseen_archives_require_a_separate_request(roots) -> None:
    source, archives, staging, cache = roots
    with pytest.raises(ValueError, match="отдельно"):
        build_tar_archives(source, archives, staging, ("train", "test_unseen"))
    build_tar_archives(source, archives, staging, ("test_unseen",), reserve_bytes=0)
    restore_tar_cache(source, archives, cache, ("test_unseen",), reserve_bytes=0)
    assert len(list((cache / "test_unseen").glob("*.flac"))) == 2
    assert not (cache / "train").exists()


def test_cpu_audit_finds_partial_and_bad_receipt(roots) -> None:
    source, archives, staging, _ = roots
    build_tar_archives(source, archives, staging, reserve_bytes=0)
    train = next(archives.glob("*/train/*.tar"))
    validation = next(archives.glob("*/validation/*.tar"))
    validation.rename(validation.with_suffix(".tar.part"))
    train.with_suffix(".json").write_text("{}", encoding="utf-8")
    report = build_tar_archives(
        source, archives, staging, reserve_bytes=0,
        plan_only=True, verify_sha=True, verify_contents=True,
    )
    assert report["planned_archives"] == 2
    assert report["verified_archives"] == 0
    assert report["partial_bytes_on_drive"] > 0
    assert report["errors"] == 1
    assert [row["status"] for row in report["archives"]] == ["Повреждён", "Незавершён"]
    build_tar_archives(source, archives, staging, reserve_bytes=0, verify_contents=True)
    report = build_tar_archives(
        source, archives, staging, reserve_bytes=0,
        plan_only=True, verify_sha=True, verify_contents=True,
    )
    assert report["verified_archives"] == 2
    assert report["pending_archives"] == 0


def test_cpu_notebook_writes_verified_report_only_after_opt_in(roots) -> None:
    source, archives, staging, cache = roots
    notebook = (
        Path(__file__).resolve().parents[1]
        / "audio_deepfake_detector/notebooks/04_ruasd_tar_archives_cpu.ipynb"
    )
    notebook_data = json.loads(notebook.read_text(encoding="utf-8"))
    assert "accelerator" not in notebook_data["metadata"]
    code = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook_data["cells"] if cell["cell_type"] == "code"
    }
    for identifier, cell in code.items():
        compile(cell, identifier, "exec")
        assert "torch" not in cell and "train_cnn" not in cell
    report_path = source.parent / "tar_archives_report.json"

    def inspect(**checks):
        return build_tar_archives(
            source, archives, staging, reserve_bytes=0, plan_only=True, **checks,
        )

    namespace = {
        "source_root": source, "archive_root": archives, "staging_root": staging,
        "options": lambda: {"splits": ("train", "validation"), "reserve_bytes": 0},
        "build_tar_archives": build_tar_archives, "inspect_archives": inspect,
        "report_path": report_path, "json": json, "show_report": lambda report: None,
    }
    exec(code["build"], namespace)  # noqa: S102
    assert not report_path.exists()
    assert not archives.exists()
    exec(code["build"].replace("RUN_TAR_BUILD = False", "RUN_TAR_BUILD = True"), namespace)  # noqa: S102
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"]
    assert report["verified_archives"] == 2
    assert restore_tar_cache(source, archives, cache, reserve_bytes=0)["restored_files"] == 4
