"""Состояние CNN-инференса без фиктивного предсказания."""

from typing import Any

from PyQt6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget


class NeuralAnalysisWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("CNN-анализ")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        form = QFormLayout()
        self.model_status = QLabel("Модель не загружена")
        self.verdict = QLabel("НЕ ВЫПОЛНЕН")
        self.verdict.setStyleSheet("font-size: 22px; font-weight: 700; color: #64748b;")
        self.confidence = QLabel("—")
        form.addRow("Состояние:", self.model_status)
        form.addRow("Вердикт:", self.verdict)
        form.addRow("Уверенность:", self.confidence)
        layout.addLayout(form)

        note = QLabel(
            "Случайный REAL/FAKE не генерируется. После обучения CNN модуль "
            "src/inference.py передаст сюда настоящий результат."
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

    def set_result(self, result: dict[str, Any]) -> None:
        if not result.get("model_loaded"):
            self.model_status.setText("Модель не загружена")
            self.verdict.setText("НЕ ВЫПОЛНЕН")
            self.confidence.setText("—")
            return
        self.model_status.setText("Модель загружена")
        verdict = str(result.get("verdict", "НЕ ВЫПОЛНЕН"))
        confidence = result.get("confidence")
        self.verdict.setText(verdict)
        self.confidence.setText(
            f"{float(confidence) * 100:.1f} %" if confidence is not None else "—"
        )

