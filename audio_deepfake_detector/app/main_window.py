"""Главное окно desktop-приложения."""

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.widgets.acoustic_analysis_widget import AcousticAnalysisWidget
from app.widgets.file_info_widget import FileInfoWidget
from app.widgets.neural_analysis_widget import NeuralAnalysisWidget
from app.widgets.result_widget import ResultWidget
from app.widgets.spectrogram_widget import SpectrogramWidget
from app.widgets.waveform_widget import WaveformWidget
from app.workers import AnalysisWorker, read_audio_metadata


class MainWindow(QMainWindow):
    """Интерфейс выбора файла и безопасного фонового анализа."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализ синтезированной речи")
        self.resize(1180, 800)
        self.setMinimumSize(900, 650)

        self.selected_file: str | None = None
        self.analysis_thread: QThread | None = None
        self.analysis_worker: AnalysisWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        root.addWidget(self._build_header())
        root.addWidget(self._build_progress_area())
        self.tabs = QTabWidget()
        self._build_tabs()
        root.addWidget(self.tabs, 1)

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("headerCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Анализ синтезированной речи")
        title.setObjectName("titleLabel")
        subtitle = QLabel(
            "Исследовательский прототип ВКР: технический и акустический анализ аудио"
        )
        subtitle.setObjectName("mutedLabel")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.file_path_label = QLabel("Аудиофайл не выбран")
        self.file_path_label.setObjectName("mutedLabel")
        self.file_path_label.setWordWrap(True)
        controls.addWidget(self.file_path_label, 1)

        self.select_button = QPushButton("Выбрать аудиофайл")
        self.select_button.clicked.connect(self.select_file)
        controls.addWidget(self.select_button)

        self.analysis_button = QPushButton("Начать анализ")
        self.analysis_button.setObjectName("primaryButton")
        self.analysis_button.setEnabled(False)
        self.analysis_button.clicked.connect(self.start_analysis)
        controls.addWidget(self.analysis_button)
        layout.addLayout(controls)
        return frame

    def _build_progress_area(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 0, 2, 0)
        self.stage_label = QLabel("Ожидание выбора файла")
        self.stage_label.setObjectName("mutedLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(420)
        layout.addWidget(self.stage_label, 1)
        layout.addWidget(self.progress_bar)
        return widget

    def _build_tabs(self) -> None:
        self.overview_info = FileInfoWidget()
        self.waveform = WaveformWidget()
        self.result_card = ResultWidget()
        self.overview_status = QLabel("Статус: анализ не выполнен")
        self.overview_status.setObjectName("mutedLabel")

        overview_content = QWidget()
        overview_layout = QVBoxLayout(overview_content)
        overview_layout.setContentsMargins(16, 16, 16, 16)
        overview_layout.addWidget(self.overview_status)
        overview_layout.addWidget(self.overview_info)
        overview_layout.addWidget(self.waveform, 1)
        overview_layout.addWidget(self.result_card)
        self.tabs.addTab(self._scrollable(overview_content), "Обзор")

        self.spectrogram = SpectrogramWidget()
        self.tabs.addTab(self.spectrogram, "Спектрограмма")

        self.neural = NeuralAnalysisWidget()
        self.tabs.addTab(self.neural, "Нейросетевой анализ")

        self.acoustic = AcousticAnalysisWidget()
        self.tabs.addTab(self.acoustic, "Акустический анализ")

        self.technical_info = FileInfoWidget()
        technical_page = QWidget()
        technical_layout = QVBoxLayout(technical_page)
        technical_layout.setContentsMargins(16, 16, 16, 16)
        technical_title = QLabel("Техническая информация")
        technical_title.setObjectName("sectionTitle")
        technical_layout.addWidget(technical_title)
        technical_layout.addWidget(self.technical_info)
        technical_layout.addStretch()
        self.tabs.addTab(self._scrollable(technical_page), "Техническая информация")

    @staticmethod
    def _scrollable(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def select_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл",
            "",
            "Аудиофайлы (*.wav *.flac *.mp3 *.ogg);;Все файлы (*)",
        )
        if not file_path:
            return

        try:
            metadata = read_audio_metadata(file_path)
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось открыть файл", str(exc))
            return

        self.selected_file = file_path
        self.file_path_label.setText(str(Path(file_path).resolve()))
        self.overview_info.set_metadata(metadata)
        self.technical_info.set_metadata(metadata)
        self.analysis_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Файл выбран — можно начать анализ")
        self.overview_status.setText("Статус: анализ не выполнен")
        self.waveform.show_placeholder()
        self.spectrogram.show_placeholder()
        self.neural.set_result({"model_loaded": False})
        self.result_card.set_result({"model_loaded": False})

    def start_analysis(self) -> None:
        if not self.selected_file:
            QMessageBox.information(self, "Файл не выбран", "Сначала выберите аудиофайл")
            return
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            return

        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.overview_status.setText("Статус: выполняется анализ…")

        self.analysis_thread = QThread(self)
        self.analysis_worker = AnalysisWorker(self.selected_file)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self.progress_bar.setValue)
        self.analysis_worker.stage_changed.connect(self.stage_label.setText)
        self.analysis_worker.finished.connect(self._analysis_finished)
        self.analysis_worker.failed.connect(self._analysis_failed)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self._thread_finished)
        self.analysis_thread.start()

    def _analysis_finished(self, result: dict[str, Any]) -> None:
        metadata = result["metadata"]
        self.overview_info.set_metadata(metadata)
        self.technical_info.set_metadata(metadata)
        self.waveform.set_audio(result["audio"], result["sample_rate"])
        self.spectrogram.set_audio(result["audio"], result["sample_rate"])
        self.acoustic.set_results(result)
        self.neural.set_result(result["cnn"])
        self.result_card.set_result(result["cnn"])
        self.overview_status.setText(
            "Статус: акустический анализ завершён; CNN-модель не загружена"
        )

        errors = result.get("errors", {})
        if errors:
            details = "\n".join(f"• {key}: {value}" for key, value in errors.items())
            QMessageBox.information(
                self,
                "Анализ завершён частично",
                "Некоторые характеристики получить не удалось:\n" + details,
            )

    def _analysis_failed(self, message: str) -> None:
        self.progress_bar.setValue(0)
        self.stage_label.setText("Ошибка анализа")
        self.overview_status.setText("Статус: анализ завершился ошибкой")
        QMessageBox.critical(self, "Ошибка анализа", message)

    def _thread_finished(self) -> None:
        self._set_busy(False)
        if self.stage_label.text() != "Ошибка анализа":
            self.stage_label.setText("Готово")
        thread = self.analysis_thread
        self.analysis_thread = None
        self.analysis_worker = None
        if thread is not None:
            thread.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        self.select_button.setEnabled(not busy)
        self.analysis_button.setEnabled(not busy and self.selected_file is not None)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — имя Qt API
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            QMessageBox.information(
                self,
                "Анализ выполняется",
                "Дождитесь завершения текущего этапа анализа перед закрытием приложения.",
            )
            event.ignore()
            return
        event.accept()

