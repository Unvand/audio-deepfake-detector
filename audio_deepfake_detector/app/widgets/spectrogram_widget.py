"""Обычная спектрограмма без выбора параметров Mel-преобразования."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from audio_deepfake_detector.app.widgets.plot_utils import show_plot_placeholder


class SpectrogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        notice = QLabel(
            "Показана обычная спектрограмма. Mel-спектрограмма будет добавлена "
            "после выбора параметров модели."
        )
        notice.setObjectName("mutedLabel")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        show_plot_placeholder(
            self.axes,
            self.canvas,
            "Спектрограмма появится после запуска анализа",
        )

    def set_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        self.axes.clear()
        max_samples = sample_rate * 120
        display_audio = audio[:max_samples]
        self.axes.specgram(
            display_audio,
            NFFT=1024,
            Fs=sample_rate,
            noverlap=512,
            cmap="magma",
        )
        suffix = " (первые 120 с)" if audio.size > max_samples else ""
        self.axes.set_title(f"Обычная спектрограмма{suffix}")
        self.axes.set_xlabel("Время, с")
        self.axes.set_ylabel("Частота, Гц")
        self.canvas.draw_idle()

