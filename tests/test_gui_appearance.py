"""Тема, сохранение настроек и локальная справка."""

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QDialog, QTextBrowser

from audio_deepfake_detector import __version__
from audio_deepfake_detector.app import main_window


def test_theme_and_help_survive_window_restart(qapp, monkeypatch, tmp_path):
    ini = str(tmp_path / "appearance.ini")
    monkeypatch.setattr(
        main_window, "QSettings", lambda *_: QSettings(ini, QSettings.Format.IniFormat)
    )
    window = main_window.MainWindow()
    window.show()
    assert "светлая" in window.theme_button.text()
    light = window.palette().color(QPalette.ColorRole.Window)
    window.theme_button.click()
    assert "тёмная" in window.theme_button.text()
    assert "светлую" in window.theme_button.toolTip()
    assert window.palette().color(QPalette.ColorRole.Window) != light
    window.waveform.canvas.draw()
    assert window.waveform.canvas.dark
    assert window.waveform.figure.get_facecolor()[0] < 0.2

    menu = window.menuBar().actions()[0].menu()
    assert menu.title() == "Справка"
    expected = {
        "Руководство пользователя": "Модель не загружена",
        "О программе": __version__,
    }
    for action in menu.actions():
        action.trigger()
        qapp.processEvents()
        dialog = next(
            dialog for dialog in window.findChildren(QDialog)
            if dialog.windowTitle() == action.text()
        )
        assert dialog.isVisible()
        assert expected[action.text()] in dialog.findChild(QTextBrowser).toPlainText()
        assert dialog.palette().color(QPalette.ColorRole.Window) != light
        dialog.close()
    assert {action.text() for action in menu.actions()} == set(expected)
    window.close()

    reopened = main_window.MainWindow()
    assert reopened.dark_theme
    reopened.theme_button.click()
    assert reopened.palette().color(QPalette.ColorRole.Window) == light
    reopened.close()
