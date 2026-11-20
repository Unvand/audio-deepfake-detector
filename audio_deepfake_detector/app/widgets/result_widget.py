"""Заметная карточка итогового результата."""

from typing import Any

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class ResultWidget(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)

        title = QLabel("Итоговый результат")
        title.setObjectName("sectionTitle")
        self.status = QLabel("МОДЕЛЬ НЕ ЗАГРУЖЕНА")
        self.status.setStyleSheet("font-size: 24px; font-weight: 750; color: #64748b;")
        self.confidence = QLabel("Уверенность: —")
        self.confidence.setObjectName("mutedLabel")
        layout.addWidget(title)
        layout.addWidget(self.status)
        layout.addWidget(self.confidence)

    def set_result(self, cnn_result: dict[str, Any]) -> None:
        if not cnn_result.get("model_loaded"):
            self.status.setText("МОДЕЛЬ НЕ ЗАГРУЖЕНА")
            self.status.setStyleSheet(
                "font-size: 24px; font-weight: 750; color: #64748b;"
            )
            self.confidence.setText("Уверенность: —")
            return

        verdict = str(cnn_result.get("verdict", "НЕ ВЫПОЛНЕН"))
        color = "#15803d" if verdict == "REAL" else "#b91c1c"
        self.status.setText(verdict)
        self.status.setStyleSheet(
            f"font-size: 28px; font-weight: 800; color: {color};"
        )
        confidence = cnn_result.get("confidence")
        text = f"{float(confidence) * 100:.1f} %" if confidence is not None else "—"
        self.confidence.setText(f"Уверенность: {text}")

