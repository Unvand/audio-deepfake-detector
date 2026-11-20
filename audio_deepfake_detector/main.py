"""Запуск desktop-приложения: ``python main.py``."""

import sys

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.styles import APP_STYLE


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Анализ синтезированной речи")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
