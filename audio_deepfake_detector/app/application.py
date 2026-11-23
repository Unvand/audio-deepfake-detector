"""Единая точка создания и запуска Qt-приложения."""

import sys
from collections.abc import Sequence

from PyQt6.QtWidgets import QApplication

from audio_deepfake_detector import __version__
from audio_deepfake_detector.app.main_window import MainWindow


def create_application(arguments: Sequence[str] | None = None) -> QApplication:
    """Создать или вернуть текущий экземпляр ``QApplication``."""
    existing = QApplication.instance()
    app = (
        existing
        if existing is not None
        else QApplication(list(arguments) if arguments is not None else sys.argv)
    )
    app.setApplicationName("Анализ синтезированной речи")
    app.setApplicationVersion(__version__)
    return app


def run_application(arguments: Sequence[str] | None = None) -> int:
    """Показать главное окно и запустить цикл обработки событий Qt."""
    app = create_application(arguments)
    window = MainWindow()
    window.show()
    return app.exec()

