"""F0, MFCC и спектральные характеристики без классификации REAL/FAKE."""

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class AcousticAnalysisWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self._create_f0_page()
        self._create_mfcc_page()
        self._create_spectral_page()

    def _create_f0_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "F0 рассчитывается через librosa.pyin и не используется как "
            "самостоятельный критерий REAL/FAKE."
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self.f0_labels: dict[str, QLabel] = {}
        for key, title in [
            ("mean", "Среднее F0"),
            ("min", "Минимальное F0"),
            ("max", "Максимальное F0"),
            ("std", "Стандартное отклонение"),
            ("voiced_ratio", "Voiced ratio"),
        ]:
            label = QLabel("—")
            self.f0_labels[key] = label
            form.addRow(f"{title}:", label)
        layout.addLayout(form)

        self.f0_figure = Figure(figsize=(8, 3), tight_layout=True)
        self.f0_canvas = FigureCanvasQTAgg(self.f0_figure)
        self.f0_axes = self.f0_figure.add_subplot(111)
        layout.addWidget(self.f0_canvas, 1)
        self.tabs.addTab(page, "F0 / Pitch")
        self._show_plot_message(self.f0_axes, self.f0_canvas, "F0 ещё не рассчитан")

    def _create_mfcc_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "MFCC показаны только как описательные признаки. Отдельный "
            "классификатор и пороги REAL/FAKE не используются."
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.mfcc_figure = Figure(figsize=(8, 3), tight_layout=True)
        self.mfcc_canvas = FigureCanvasQTAgg(self.mfcc_figure)
        self.mfcc_axes = self.mfcc_figure.add_subplot(111)
        layout.addWidget(self.mfcc_canvas, 2)

        self.mfcc_table = QTableWidget(0, 3)
        self.mfcc_table.setHorizontalHeaderLabels(["Коэффициент", "Среднее", "Std"])
        self.mfcc_table.verticalHeader().setVisible(False)
        self.mfcc_table.setMaximumHeight(230)
        layout.addWidget(self.mfcc_table, 1)
        self.tabs.addTab(page, "MFCC")
        self._show_plot_message(
            self.mfcc_axes, self.mfcc_canvas, "MFCC ещё не рассчитаны"
        )

    def _create_spectral_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Характеристики описывают спектр и не формируют вердикт REAL/FAKE."
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self.spectral_labels: dict[str, QLabel] = {}
        fields = [
            ("spectral_centroid_hz", "Spectral centroid"),
            ("spectral_bandwidth_hz", "Spectral bandwidth"),
            ("spectral_rolloff_hz", "Spectral rolloff"),
            ("zero_crossing_rate", "Zero crossing rate"),
        ]
        for key, title in fields:
            label = QLabel("—")
            self.spectral_labels[key] = label
            form.addRow(f"{title}:", label)
        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(page, "Спектральный анализ")

    @staticmethod
    def _show_plot_message(axes: Any, canvas: Any, message: str) -> None:
        axes.clear()
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#64748b",
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        canvas.draw_idle()

    def set_results(self, result: dict[str, Any]) -> None:
        self._set_f0(result.get("f0"), result.get("errors", {}).get("f0"))
        self._set_mfcc(result.get("mfcc"), result.get("errors", {}).get("mfcc"))
        self._set_spectral(
            result.get("spectral"), result.get("errors", {}).get("spectral")
        )

    def _set_f0(self, data: dict[str, Any] | None, error: str | None) -> None:
        if data is None:
            for label in self.f0_labels.values():
                label.setText("Недоступно")
            self._show_plot_message(
                self.f0_axes,
                self.f0_canvas,
                f"F0 недоступен: {error or 'нет данных'}",
            )
            return

        for key in ("mean", "min", "max", "std"):
            self.f0_labels[key].setText(f"{data[key]:.2f} Гц")
        self.f0_labels["voiced_ratio"].setText(f"{data['voiced_ratio'] * 100:.1f} %")

        self.f0_axes.clear()
        self.f0_axes.plot(data["times"], data["values"], color="#7c3aed", linewidth=1)
        self.f0_axes.set_title("Изменение F0 во времени")
        self.f0_axes.set_xlabel("Время, с")
        self.f0_axes.set_ylabel("F0, Гц")
        self.f0_axes.grid(alpha=0.2)
        self.f0_canvas.draw_idle()

    def _set_mfcc(self, data: dict[str, Any] | None, error: str | None) -> None:
        if data is None:
            self.mfcc_table.setRowCount(0)
            self._show_plot_message(
                self.mfcc_axes,
                self.mfcc_canvas,
                f"MFCC недоступны: {error or 'нет данных'}",
            )
            return

        values = np.asarray(data["values"])
        self.mfcc_axes.clear()
        self.mfcc_axes.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        self.mfcc_axes.set_title("MFCC во времени")
        self.mfcc_axes.set_xlabel("Кадр")
        self.mfcc_axes.set_ylabel("Номер коэффициента")
        self.mfcc_canvas.draw_idle()

        means = np.asarray(data["mean"])
        stds = np.asarray(data["std"])
        self.mfcc_table.setRowCount(len(means))
        for index, (mean, std) in enumerate(zip(means, stds, strict=True)):
            self.mfcc_table.setItem(index, 0, QTableWidgetItem(f"MFCC {index + 1}"))
            self.mfcc_table.setItem(index, 1, QTableWidgetItem(f"{mean:.4f}"))
            self.mfcc_table.setItem(index, 2, QTableWidgetItem(f"{std:.4f}"))
        self.mfcc_table.resizeColumnsToContents()

    def _set_spectral(
        self, data: dict[str, float] | None, error: str | None
    ) -> None:
        if data is None:
            for label in self.spectral_labels.values():
                label.setText(f"Недоступно ({error or 'нет данных'})")
            return
        for key, label in self.spectral_labels.items():
            suffix = " Гц" if key != "zero_crossing_rate" else ""
            label.setText(f"{data[key]:.4f}{suffix}")

