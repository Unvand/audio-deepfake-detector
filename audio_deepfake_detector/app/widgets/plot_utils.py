"""Общие операции для встроенных Matplotlib-графиков."""

from typing import Protocol

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class ThemedCanvas(FigureCanvasQTAgg):
    """Применить палитру к графику, в том числе после очистки осей."""

    dark = False

    def draw(self) -> None:
        background, foreground, muted = (
            ("#242b36", "#e4e9f2", "#aab7cb") if self.dark
            else ("#ffffff", "#1f2937", "#64748b")
        )
        self.figure.set_facecolor(background)
        line_colors = {"#2563eb": "#8ab4ff", "#7c3aed": "#c4a4ff"}
        if not self.dark:
            line_colors = {value: key for key, value in line_colors.items()}
        for axes in self.figure.axes:
            axes.set_facecolor(background)
            axes.tick_params(colors=muted)
            axes.xaxis.label.set_color(foreground)
            axes.yaxis.label.set_color(foreground)
            axes.title.set_color(foreground)
            for spine in axes.spines.values():
                spine.set_color(muted)
            for text in axes.texts:
                text.set_color(muted)
            for line in axes.lines:
                color = line.get_color()
                if isinstance(color, str) and color in line_colors:
                    line.set_color(line_colors[color])
        super().draw()


class DrawableCanvas(Protocol):
    """Минимальный интерфейс canvas, нужный для перерисовки."""

    def draw_idle(self) -> None: ...


def show_plot_placeholder(axes: Axes, canvas: DrawableCanvas, message: str) -> None:
    """Очистить график и показать нейтральное текстовое сообщение."""
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

