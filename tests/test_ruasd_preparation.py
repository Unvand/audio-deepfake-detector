"""Тесты подготовки RuASD без сети и Hugging Face."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audio_deepfake_detector.src.ruasd_preparation import (
    DecodedAudio,
    PreparationConfig,
    decoder_to_mono,
    prepare_ruasd_dataset,
    segment_with_padding,
    validate_ruasd_manifest,
    verify_prepared_dataset,
)


class _DecodedSamples:
    def __init__(self) -> None:
        self.data = np.vstack(
            [np.ones(16, dtype=np.float32), np.zeros(16, dtype=np.float32)]
        )
        self.sample_rate = 48_000


class _StereoDecoder:
    def get_all_samples(self) -> _DecodedSamples:
        return _DecodedSamples()


def _manifest() -> pd.DataFrame:
    rows = []
    for index, (label, split) in enumerate(
        [
            ("real", "train"),
            ("real", "train"),
            ("fake", "train"),
            ("fake", "train"),
            ("real", "validation"),
            ("fake", "validation"),
            ("real", "test_unseen"),
            ("fake", "test_unseen"),
        ]
    ):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "label": label,
                "split": split,
                "subset": "source",
                "model": "",
                "duration": -1.0 if index == 0 else 0.25,
                "shard_id": index % 2,
                "__key__": f"sample-{index}",
                "__url__": f"hf://shard-{index % 2}",
            }
        )
    return pd.DataFrame(rows)


class _FakeReader:
    calls = 0

    def __init__(self, token: str, config: PreparationConfig) -> None:
        assert token == "test-token"
        self.config = config

    def iter_decoded(self, records: pd.DataFrame):
        type(self).calls += 1
        for row in records.to_dict("records"):
            # Stereo-to-mono уже покрывается decoder_to_mono отдельно; здесь
            # проверяем весь transactional pipeline на mono 16 кГц.
            waveform = np.linspace(-0.2, 0.2, 4_001, dtype=np.float32)
            yield row, DecodedAudio(waveform, 16_000), None


def test_manifest_validation_matches_fixed_split() -> None:
    summary = validate_ruasd_manifest(_manifest())
    assert summary["records"] == 8
    assert summary["unknown_duration"] == 1
    assert summary["unique_sample_ids"] == 8


def test_segment_padding_preserves_valid_duration() -> None:
    waveform = np.ones(10, dtype=np.float32)
    segments = segment_with_padding(waveform, sample_rate=4, segment_seconds=2.0)
    assert [len(segment) for segment, _ in segments] == [8, 8]
    assert [duration for _, duration in segments] == [2.0, 0.5]
    assert np.all(segments[-1][0][2:] == 0)


def test_audio_decoder_converts_stereo_to_mono() -> None:
    decoded = decoder_to_mono(_StereoDecoder())
    assert decoded.sample_rate == 48_000
    assert decoded.waveform.shape == (16,)
    np.testing.assert_allclose(decoded.waveform, 0.5)


def test_prepare_flac_and_resume_without_duplicates(tmp_path: Path) -> None:
    _FakeReader.calls = 0
    config = PreparationConfig(target_sample_rate=32_000, segment_seconds=0.25)
    first = prepare_ruasd_dataset(
        _manifest(),
        tmp_path,
        "test-token",
        config,
        full_mode=False,
        test_limit=4,
        checkpoint_every=1,
        reader_factory=_FakeReader,
    )
    segments = pd.read_csv(tmp_path / "ruasd_segments_manifest.csv")
    assert first["processed_this_run"] == 4
    assert segments["sample_id"].nunique() == 4
    assert not segments["segment_id"].duplicated().any()
    for relative_path in segments["audio_path"]:
        info = sf.info(tmp_path / relative_path)
        assert info.samplerate == 32_000
        assert info.frames == 8_000
        assert info.format == "FLAC"

    second = prepare_ruasd_dataset(
        _manifest(),
        tmp_path,
        "test-token",
        config,
        full_mode=False,
        test_limit=4,
        reader_factory=_FakeReader,
    )
    resumed_segments = pd.read_csv(tmp_path / "ruasd_segments_manifest.csv")
    assert second["processed_this_run"] == 0
    assert len(resumed_segments) == len(segments)
    assert _FakeReader.calls == 1
    assert verify_prepared_dataset(tmp_path, 32_000, 0.25)["valid"] is True
