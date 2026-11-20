"""Единый стиль PyQt6-приложения."""

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
"""
