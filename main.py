"""Корневая точка запуска после клонирования репозитория."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "audio_deepfake_detector"
sys.path.insert(0, str(PROJECT_DIR))

from app.main_window import MainWindow  # noqa: E402
from app.styles import APP_STYLE  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Анализ синтезированной речи")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

