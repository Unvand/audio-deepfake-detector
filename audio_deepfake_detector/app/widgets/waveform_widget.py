"""Визуализация временной формы сигнала."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from audio_deepfake_detector.app.widgets.plot_utils import show_plot_placeholder


class WaveformWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        show_plot_placeholder(
            self.axes,
            self.canvas,
            "Waveform появится после запуска анализа",
        )

    def set_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        self.axes.clear()
        step = max(1, int(np.ceil(audio.size / 50_000)))
        display_audio = audio[::step]
        times = np.arange(display_audio.size) * step / sample_rate
        self.axes.plot(times, display_audio, color="#2563eb", linewidth=0.7)
        self.axes.set_title("Временная форма аудиосигнала")
        self.axes.set_xlabel("Время, с")
        self.axes.set_ylabel("Амплитуда")
        self.axes.grid(alpha=0.2)
        self.canvas.draw_idle()

