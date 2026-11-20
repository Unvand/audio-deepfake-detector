"""Обычная спектрограмма без выбора параметров Mel-преобразования."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


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
        self.axes.clear()
        self.axes.text(
            0.5,
            0.5,
            "Спектрограмма появится после запуска анализа",
            ha="center",
            va="center",
            color="#64748b",
            transform=self.axes.transAxes,
        )
        self.axes.set_axis_off()
        self.canvas.draw_idle()

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

