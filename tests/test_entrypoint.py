"""Smoke-тест корневой точки запуска и структуры пакетов."""

from audio_deepfake_detector.app.application import create_application
from audio_deepfake_detector.app.main_window import MainWindow


def test_application_can_be_created_from_repository_root(qapp: object) -> None:
    app = create_application([])
    window = MainWindow()

    assert app.applicationName() == "Анализ синтезированной речи"
    assert window.windowTitle() == "Анализ синтезированной речи"
    window.close()

