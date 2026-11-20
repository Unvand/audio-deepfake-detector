"""Общие настройки тестов, включая headless-запуск Qt."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "audio_deepfake_detector_mpl"),
)


@pytest.fixture(scope="session")
def qapp() -> Iterator[object]:
    """Предоставить единственный ``QApplication`` для GUI-тестов."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()

