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
        self.status.setObjectName("resultStatus")
        self.confidence = QLabel("Уверенность: —")
        self.confidence.setObjectName("mutedLabel")
        layout.addWidget(title)
        layout.addWidget(self.status)
        layout.addWidget(self.confidence)

    def set_result(self, cnn_result: dict[str, Any]) -> None:
        if not cnn_result.get("model_loaded"):
            self.status.setText("МОДЕЛЬ НЕ ЗАГРУЖЕНА")
            self._set_verdict_color("")
            self.confidence.setText("Уверенность: —")
            return

        verdict = str(cnn_result.get("verdict", "НЕ ВЫПОЛНЕН"))
        self.status.setText(verdict)
        self._set_verdict_color(verdict)
        confidence = cnn_result.get("confidence")
        text = f"{float(confidence) * 100:.1f} %" if confidence is not None else "—"
        self.confidence.setText(f"Уверенность: {text}")

    def _set_verdict_color(self, verdict: str) -> None:
        self.status.setProperty("verdict", verdict)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

