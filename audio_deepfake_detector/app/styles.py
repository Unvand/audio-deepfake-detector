"""Единый стиль PyQt6-приложения."""

import re

from PyQt6.QtGui import QColor, QPalette

APP_STYLE = """
QMainWindow, QWidget {
    background-color: #f4f6f9;
    color: #1f2937;
    font-family: "Segoe UI", "Arial";
    font-size: 13px;
}
QFrame#headerCard, QFrame#contentCard, QFrame#resultCard {
    background-color: #ffffff;
    border: 1px solid #dbe2ea;
    border-radius: 10px;
}
QLabel#titleLabel {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
}
QLabel#sectionTitle {
    font-size: 17px;
    font-weight: 650;
    color: #172033;
}
QLabel#mutedLabel {
    color: #64748b;
}
QPushButton {
    min-height: 36px;
    padding: 0 16px;
    border-radius: 7px;
    border: 1px solid #cbd5e1;
    background-color: #ffffff;
    font-weight: 600;
}
QPushButton:hover { background-color: #eef4ff; border-color: #6b9df5; }
QPushButton#primaryButton {
    background-color: #2563eb;
    color: white;
    border-color: #2563eb;
}
QPushButton#primaryButton:hover { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #e5e7eb; color: #94a3b8; border-color: #e5e7eb; }
QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #dbe2ea;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background: #e9eef5;
    padding: 10px 16px;
    margin-right: 2px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}
QTabBar::tab:selected { background: #ffffff; color: #1d4ed8; font-weight: 650; }
QProgressBar {
    min-height: 18px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    background: #e9eef5;
    text-align: center;
}
QProgressBar::chunk { background-color: #2563eb; border-radius: 6px; }
QGroupBox {
    font-weight: 650;
    border: 1px solid #dbe2ea;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QMenuBar::item:selected, QMenu::item:selected { background: #e9eef5; }
QToolTip { background: #ffffff; color: #1f2937; border: 1px solid #cbd5e1; }
QLabel#neuralVerdict { font-size: 22px; font-weight: 700; color: #64748b; }
QLabel#resultStatus { font-size: 24px; font-weight: 750; color: #64748b; }
QLabel#resultStatus[verdict="REAL"] { font-size: 28px; color: #15803d; }
QLabel#resultStatus[verdict="FAKE"] { font-size: 28px; color: #b91c1c; }
"""

DARK_COLORS = {
    "#f4f6f9": "#1b2029", "#1f2937": "#e4e9f2", "#ffffff": "#242b36",
    "#dbe2ea": "#414c5e", "#111827": "#f3f6fc", "#172033": "#edf1f8",
    "#64748b": "#aab7cb", "#cbd5e1": "#526077", "#eef4ff": "#34445c",
    "#e5e7eb": "#303947", "#94a3b8": "#a0adc1", "#e9eef5": "#303c50",
    "#15803d": "#81d8a0", "#b91c1c": "#ffa8a8",
}


def theme_style(dark: bool) -> str:
    """Вернуть общий набор правил в нужной палитре."""
    if dark:
        style = APP_STYLE.replace("; color: #1d4ed8;", "; color: #a8c7ff;")
        return re.sub(r"#[0-9a-f]{6}\b", lambda match: DARK_COLORS.get(match[0], match[0]), style)
    return APP_STYLE


def theme_palette(dark: bool, base: QPalette) -> QPalette:
    """Настроить цвета нативных полей, таблиц, меню и выпадающих списков."""
    palette = QPalette(base)
    role = QPalette.ColorRole
    for key, color in {
        role.Window: "#f4f6f9", role.WindowText: "#1f2937", role.Base: "#ffffff",
        role.AlternateBase: "#e9eef5", role.Text: "#1f2937", role.Button: "#ffffff",
        role.ButtonText: "#1f2937", role.ToolTipBase: "#ffffff", role.ToolTipText: "#1f2937",
        role.Highlight: "#2563eb", role.HighlightedText: "white",
    }.items():
        palette.setColor(key, QColor(DARK_COLORS.get(color, color) if dark else color))
    for key in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, key, QColor("#94a3b8" if dark else "#64748b"))
    return palette
