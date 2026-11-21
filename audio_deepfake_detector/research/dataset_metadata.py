"""Безопасная загрузка и анализ открытых metadata/protocol файлов.

Модуль намеренно не загружает аудиокорпуса автоматически. Любой файл,
размер которого превышает ``max_bytes``, требует явного ``confirm_large=True``.
Это особенно важно в Google Colab, где объём временного диска ограничен.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MIB = 1024**2
DEFAULT_SMALL_DOWNLOAD_LIMIT = 64 * MIB

ASVSPOOF5_COLUMNS = [
    "speaker_id",
    "file_name",
    "gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
    "tmp",
]

ASVSPOOF2021_DF_COLUMNS = [
    "speaker",
    "trial",
    "codec",
    "source",
    "attack",
    "label",
    "trim",
    "subset",
    "vocoder",
    "task",
    "team",
    "gender_pair",
    "language",
]


@dataclass(frozen=True)
class DownloadSpec:
    """Описание официального файла и его ожидаемого размера."""

    name: str
    url: str
    expected_bytes: int | None
    metadata_only: bool


DATASET_DOWNLOADS: Mapping[str, DownloadSpec] = {
    "asvspoof5_protocols": DownloadSpec(
        name="ASVspoof 5 protocols",
        url=(
            "https://zenodo.org/api/records/14498691/files/"
            "ASVspoof5_protocols.tar.gz/content"
        ),
        expected_bytes=20_669_392,
        metadata_only=True,
    ),
    "asvspoof2021_df_keys": DownloadSpec(
        name="ASVspoof 2021 DF keys and metadata",
        url="https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz",
        expected_bytes=25_985_925,
        metadata_only=True,
    ),
    "asvspoof5_train_audio": DownloadSpec(
        name="ASVspoof 5 train audio (5 tar files)",
        url="https://zenodo.org/records/14498691",
        expected_bytes=37_501_552_640,
        metadata_only=False,
    ),
    "asvspoof5_dev_audio": DownloadSpec(
        name="ASVspoof 5 dev audio (3 tar files)",
        url="https://zenodo.org/records/14498691",
        expected_bytes=20_060_385_280,
        metadata_only=False,
    ),
    "asvspoof5_eval_audio": DownloadSpec(
        name="ASVspoof 5 eval audio (10 tar files)",
        url="https://zenodo.org/records/14498691",
        expected_bytes=84_757_227_520,
        metadata_only=False,
    ),
    "asvspoof2019_la": DownloadSpec(
        name="ASVspoof 2019 LA",
        url=(
            "https://datashare.ed.ac.uk/bitstreams/"
            "a9f87c35-f055-4015-80e2-2fdff0d46269/download"
        ),
        expected_bytes=7_640_952_520,
        metadata_only=False,
    ),
    "asvspoof2021_df": DownloadSpec(
        name="ASVspoof 2021 DF evaluation audio",
        url="https://zenodo.org/records/4835108",
        expected_bytes=34_534_483_584,
        metadata_only=False,
    ),
    "wavefake": DownloadSpec(
        name="WaveFake generated audio",
        url="https://zenodo.org/records/5642694",
        expected_bytes=28_918_626_084,
        metadata_only=False,
    ),
    "in_the_wild": DownloadSpec(
        name="In-the-Wild",
        url=(
            "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/"
            "main/release_in_the_wild.zip"
        ),
        expected_bytes=8_161_489_023,
        metadata_only=False,
    ),
    "spoofceleb": DownloadSpec(
        name="SpoofCeleb",
        url="https://huggingface.co/datasets/jungjee/spoofceleb",
        expected_bytes=134_157_296_519,
        metadata_only=False,
    ),
    "ruasd": DownloadSpec(
        name="RuASD",
        url="https://huggingface.co/datasets/lab260/RuASD",
        expected_bytes=250_452_976_559,
        metadata_only=False,
    ),
}


class LargeDownloadConfirmationRequired(RuntimeError):
    """Файл превышает безопасный лимит и требует явного подтверждения."""


def format_bytes(value: int | None) -> str:
    """Вернуть читаемое представление размера в двоичных единицах."""

    if value is None:
        return "UNKNOWN"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def _request(url: str, *, method: str = "GET", token: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": "audio-deepfake-detector-dataset-research/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers, method=method)


def remote_file_size(url: str, *, token: str | None = None, timeout: int = 30) -> int | None:
    """Получить Content-Length без скачивания тела, если сервер его сообщает."""

    request = _request(url, method="HEAD", token=token)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def download_file(
    url: str,
    destination: str | Path,
    *,
    expected_bytes: int | None = None,
    max_bytes: int = DEFAULT_SMALL_DOWNLOAD_LIMIT,
    confirm_large: bool = False,
    token: str | None = None,
    timeout: int = 120,
) -> Path:
    """Скачать файл с предохранителем от неявной многогигабайтной загрузки.

    ``expected_bytes`` предпочтителен для серверов, которые не поддерживают
    HEAD. При отсутствии оценки функция пытается прочитать Content-Length.
    """

    destination = Path(destination)
    known_size = expected_bytes
    if known_size is None:
        try:
            known_size = remote_file_size(url, token=token, timeout=min(timeout, 30))
        except OSError:
            known_size = None

    if known_size is None and not confirm_large:
        raise LargeDownloadConfirmationRequired(
            "Сервер не сообщил размер. Проверьте источник и повторите с "
            "confirm_large=True только после осознанного подтверждения."
        )
    if known_size is not None and known_size > max_bytes and not confirm_large:
        raise LargeDownloadConfirmationRequired(
            f"Запрошено {format_bytes(known_size)}; безопасный лимит "
            f"{format_bytes(max_bytes)}. Для большой загрузки задайте "
            "confirm_large=True."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = _request(url, token=token)
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    return destination


def extract_tar_safely(archive: str | Path, destination: str | Path) -> list[Path]:
    """Распаковать tar, запрещая выход путей за каталог назначения."""

    archive = Path(archive)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Недопустимый тип записи в архиве: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Небезопасный путь в архиве: {member.name}")
        bundle.extractall(destination)
        extracted = [destination / member.name for member in bundle.getmembers()]
    return extracted


def load_asvspoof5_protocol(path: str | Path) -> pd.DataFrame:
    """Прочитать официальный 10-польный protocol Track 1 ASVspoof 5."""

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=ASVSPOOF5_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    if frame.shape[1] != len(ASVSPOOF5_COLUMNS):
        raise ValueError("Неожиданная структура протокола ASVspoof 5")
    invalid_keys = set(frame["key"]) - {"bonafide", "spoof"}
    if invalid_keys:
        raise ValueError(f"Неизвестные значения KEY: {sorted(invalid_keys)}")
    return frame


def analyse_asvspoof5(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Сформировать воспроизводимые сводки по разделам ASVspoof 5."""

    split_rows: list[dict[str, Any]] = []
    attack_rows: list[pd.DataFrame] = []
    codec_rows: list[pd.DataFrame] = []
    for split, frame in frames.items():
        counts = frame["key"].value_counts()
        split_rows.append(
            {
                "split": split,
                "files": len(frame),
                "bonafide": int(counts.get("bonafide", 0)),
                "spoof": int(counts.get("spoof", 0)),
                "speakers": int(frame["speaker_id"].nunique()),
            }
        )
        attacks = (
            frame.loc[frame["key"].eq("spoof"), "attack_label"]
            .value_counts()
            .rename_axis("attack_label")
            .reset_index(name="files")
        )
        attacks.insert(0, "split", split)
        attack_rows.append(attacks)
        codecs = frame["codec"].value_counts().rename_axis("codec").reset_index(name="files")
        codecs.insert(0, "split", split)
        codec_rows.append(codecs)

    return {
        "splits": pd.DataFrame(split_rows),
        "attacks": pd.concat(attack_rows, ignore_index=True),
        "codecs": pd.concat(codec_rows, ignore_index=True),
    }


def load_asvspoof2021_df_metadata(path: str | Path) -> pd.DataFrame:
    """Прочитать официальный 13-польный metadata-файл ASVspoof 2021 DF."""

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=ASVSPOOF2021_DF_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    invalid_labels = set(frame["label"]) - {"bonafide", "spoof"}
    if invalid_labels:
        raise ValueError(f"Неизвестные значения label: {sorted(invalid_labels)}")
    return frame


def fetch_huggingface_tree(
    repository: str,
    *,
    subpath: str = "",
    token: str | None = None,
    limit: int = 1000,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Получить публичный список файлов HF без скачивания аудио."""

    encoded_path = "/" + urllib.parse.quote(subpath.strip("/"), safe="/") if subpath else ""
    url = (
        f"https://huggingface.co/api/datasets/{repository}/tree/main{encoded_path}"
        f"?recursive=true&expand=false&limit={limit}"
    )
    with urllib.request.urlopen(_request(url, token=token), timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise TypeError("Hugging Face вернул неожиданную структуру ответа")
    return payload


def fetch_ruasd_preview(*, length: int = 20, timeout: int = 30) -> pd.DataFrame:
    """Получить небольшую metadata-выборку RuASD через Dataset Viewer.

    Аудио и временные URL ресурсов намеренно не возвращаются.
    """

    if not 1 <= length <= 100:
        raise ValueError("length должен быть от 1 до 100")
    query = urllib.parse.urlencode(
        {
            "dataset": "lab260/RuASD",
            "config": "default",
            "split": "train",
        }
    )
    url = f"https://datasets-server.huggingface.co/first-rows?{query}"
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        payload = json.load(response)
    records = []
    for item in payload.get("rows", [])[:length]:
        metadata = item.get("row", {}).get("json", {})
        if isinstance(metadata, dict):
            records.append(metadata)
    return pd.DataFrame.from_records(records)


def describe_metadata(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Посчитать распределения только для реально присутствующих полей."""

    result: dict[str, pd.DataFrame] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        result[column] = (
            frame[column]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="count")
        )
    return result
