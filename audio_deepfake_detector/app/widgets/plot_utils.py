"""Общие операции для встроенных Matplotlib-графиков."""

from typing import Protocol

from matplotlib.axes import Axes


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

