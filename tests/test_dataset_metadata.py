from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from audio_deepfake_detector.research.dataset_metadata import (
    LargeDownloadConfirmationRequired,
    analyse_asvspoof5,
    download_file,
    extract_tar_safely,
    format_bytes,
    load_asvspoof5_protocol,
    load_asvspoof2021_df_metadata,
)


def test_format_bytes() -> None:
    assert format_bytes(None) == "UNKNOWN"
    assert format_bytes(20_669_392) == "19.71 MiB"


def test_large_download_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(LargeDownloadConfirmationRequired):
        download_file(
            "https://example.invalid/large.tar",
            tmp_path / "large.tar",
            expected_bytes=10_000,
            max_bytes=100,
        )


def test_asvspoof5_protocol_and_summary(tmp_path: Path) -> None:
    protocol = tmp_path / "ASVspoof5.train.tsv"
    protocol.write_text(
        "T_0001 T_0000000001 F - - - AC2 A01 spoof -\n"
        "T_0002 T_0000000002 M - - - - bonafide bonafide -\n",
        encoding="utf-8",
    )

    frame = load_asvspoof5_protocol(protocol)
    summary = analyse_asvspoof5({"train": frame})

    assert list(frame.columns)[0:3] == ["speaker_id", "file_name", "gender"]
    assert summary["splits"].iloc[0].to_dict() == {
        "split": "train",
        "files": 2,
        "bonafide": 1,
        "spoof": 1,
        "speakers": 2,
    }
    assert summary["attacks"].iloc[0]["attack_label"] == "A01"


def test_asvspoof2021_df_metadata_columns(tmp_path: Path) -> None:
    metadata = tmp_path / "trial_metadata.txt"
    metadata.write_text(
        "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim "
        "progress traditional_vocoder - - - -\n",
        encoding="utf-8",
    )

    frame = load_asvspoof2021_df_metadata(metadata)

    assert frame.loc[0, "trial"] == "DF_E_2000011"
    assert frame.loc[0, "codec"] == "nocodec"
    assert frame.loc[0, "vocoder"] == "traditional_vocoder"


def test_safe_tar_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        data = b"unsafe"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(data)
        bundle.addfile(member, io.BytesIO(data))

    with pytest.raises(ValueError, match="Небезопасный путь"):
        extract_tar_safely(archive, tmp_path / "output")
