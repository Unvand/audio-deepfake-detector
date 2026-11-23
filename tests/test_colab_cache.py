"""Тесты частичного и возобновляемого локального Colab-кэша."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from audio_deepfake_detector.src import colab_cache
from audio_deepfake_detector.src.colab_cache import (
    MANIFEST_NAME,
    cache_space_report,
    check_local_cache,
    prepare_local_cache,
    require_ready_cache,
)


def _source_dataset(root: Path) -> pd.DataFrame:
    rows = []
    index = 0
    for split in ["train", "validation", "test_unseen"]:
        for label in ["real", "fake"]:
            relative = Path(split) / f"segment-{index}.flac"
            (root / split).mkdir(parents=True, exist_ok=True)
            waveform = np.linspace(-0.1, 0.1, 3_200, dtype=np.float32)
            sf.write(root / relative, waveform, 32_000, format="FLAC", subtype="PCM_16")
            rows.append(
                {
                    "segment_id": f"segment-{index}",
                    "sample_id": f"sample-{index}",
                    "label": label,
                    "split": split,
                    "subset": "source",
                    "audio_path": relative.as_posix(),
                }
            )
            index += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(root / MANIFEST_NAME, index=False)
    return frame


def test_partial_cache_creation_and_repeat_skips_files(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    cache = tmp_path / "content_cache"
    manifest = _source_dataset(source)
    updates: list[tuple[int, int, dict[str, int]]] = []
    result = prepare_local_cache(
        source,
        cache,
        ("train", "validation"),
        reserve_bytes=0,
        checkpoint_every=1,
        progress_callback=lambda current, total, stats: updates.append(
            (current, total, stats.copy())
        ),
    )
    selected = manifest[manifest["split"].isin(["train", "validation"])]
    assert result.ready is True
    assert result.copied_files == len(selected)
    assert result.skipped_files == 0
    assert updates[-1][0:2] == (len(selected), len(selected))
    assert (cache / MANIFEST_NAME).read_bytes() == (source / MANIFEST_NAME).read_bytes()
    for relative in selected["audio_path"]:
        assert (cache / relative).is_file()
        assert not (cache / relative).is_symlink()
    unseen = manifest[manifest["split"] == "test_unseen"]
    assert all(not (cache / relative).exists() for relative in unseen["audio_path"])

    repeated = prepare_local_cache(
        source, cache, ("train", "validation"), reserve_bytes=0
    )
    assert repeated.ready is True
    assert repeated.copied_files == 0
    assert repeated.skipped_files == len(selected)


def test_missing_and_corrupted_files_are_repaired(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    cache = tmp_path / "content_cache"
    manifest = _source_dataset(source)
    prepare_local_cache(source, cache, ("train", "validation"), reserve_bytes=0)
    selected_paths = manifest[
        manifest["split"].isin(["train", "validation"])
    ]["audio_path"].tolist()
    missing_path = cache / selected_paths[0]
    corrupted_path = cache / selected_paths[1]
    missing_path.unlink()
    original_size = corrupted_path.stat().st_size
    corrupted_path.write_bytes(b"x" * original_size)

    status = check_local_cache(source, cache, ("train", "validation"))
    assert status.ready is False
    assert status.missing_files == 1
    assert status.corrupted_files == 1

    repaired = prepare_local_cache(
        source, cache, ("train", "validation"), reserve_bytes=0
    )
    assert repaired.ready is True
    assert repaired.copied_files == 2
    assert repaired.skipped_files == len(selected_paths) - 2
    assert check_local_cache(source, cache, ("train", "validation")).ready is True


def test_cache_can_be_extended_with_test_unseen(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    cache = tmp_path / "content_cache"
    manifest = _source_dataset(source)
    prepare_local_cache(source, cache, ("train", "validation"), reserve_bytes=0)
    test_result = prepare_local_cache(
        source, cache, ("test_unseen",), reserve_bytes=0
    )
    assert test_result.copied_files == 2
    assert test_result.ready is True
    assert check_local_cache(source, cache, ("test_unseen",)).ready is True
    assert check_local_cache(
        source, cache, ("train", "validation", "test_unseen")
    ).ready is True
    assert len(pd.read_csv(cache / MANIFEST_NAME)) == len(manifest)


def test_space_report_and_required_cache_path(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    cache = tmp_path / "content_cache"
    _source_dataset(source)
    report = cache_space_report(
        source, cache, ("train", "validation"), reserve_bytes=0
    )
    assert report["selected_files"] == 4
    assert report["additional_bytes"] > 0
    assert report["enough_space"] is True
    prepare_local_cache(source, cache, ("train", "validation"), reserve_bytes=0)
    assert require_ready_cache(source, cache) == cache


def test_insufficient_space_stops_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "drive"
    cache = tmp_path / "content_cache"
    _source_dataset(source)
    monkeypatch.setattr(
        colab_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )
    with pytest.raises(OSError, match="Недостаточно места"):
        prepare_local_cache(source, cache, ("train", "validation"), reserve_bytes=0)
    assert not list(cache.rglob("*.flac"))
