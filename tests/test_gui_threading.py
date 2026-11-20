"""GUI-тесты отзывчивости и безопасного завершения QThread."""

import time
from threading import Event
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from audio_deepfake_detector.app import main_window


def _wait_until(qapp: Any, condition: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class SlowCancellableWorker(QObject):
    """Управляемый test double с тем же сигнальным контрактом."""

    progress = pyqtSignal(int)
    stage_changed = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self.cancel_requested = Event()

    def cancel(self) -> None:
        self.cancel_requested.set()

    @pyqtSlot()
    def run(self) -> None:
        self.stage_changed.emit("Тестовый длительный этап")
        for _ in range(50):
            if self.cancel_requested.is_set():
                self.cancelled.emit()
                return
            time.sleep(0.005)
        self.cancelled.emit()


def test_analysis_runs_without_blocking_gui_event_loop(qapp: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(main_window, "AnalysisWorker", SlowCancellableWorker)
    window = main_window.MainWindow()
    window.selected_file = "unused.wav"
    window.analysis_button.setEnabled(True)
    timer_fired: list[bool] = []

    window.start_analysis()
    QTimer.singleShot(20, lambda: timer_fired.append(True))

    assert _wait_until(qapp, lambda: bool(timer_fired), timeout=1.0)
    assert window.analysis_thread is not None
    assert window.analysis_thread.isRunning()
    assert _wait_until(qapp, lambda: window.analysis_thread is None, timeout=2.0)
    window.close()


def test_close_requests_worker_cancel_and_finishes_thread(
    qapp: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(main_window, "AnalysisWorker", SlowCancellableWorker)
    window = main_window.MainWindow()
    window.selected_file = "unused.wav"
    window.analysis_button.setEnabled(True)
    window.show()
    window.start_analysis()
    assert _wait_until(
        qapp,
        lambda: window.analysis_thread is not None and window.analysis_thread.isRunning(),
    )
    worker = window.analysis_worker

    window.close()

    assert worker is not None
    assert worker.cancel_requested.is_set()
    assert _wait_until(qapp, lambda: window.analysis_thread is None, timeout=2.0)
    assert _wait_until(qapp, lambda: not window.isVisible(), timeout=1.0)

