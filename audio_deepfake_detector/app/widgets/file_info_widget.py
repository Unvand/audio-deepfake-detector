"""Отображение технических сведений об аудиофайле."""

from typing import Any

from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "Недоступно"
    size = float(size_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} Б"


class FileInfoWidget(QWidget):
    """Таблица с основной и расширенной информацией о файле."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(9)

        fields = [
            ("name", "Имя файла"),
            ("path", "Путь"),
            ("duration", "Длительность"),
            ("sample_rate", "Sample rate"),
            ("channels", "Каналы"),
            ("format", "Формат / codec"),
            ("subtype", "Подтип"),
            ("bitrate", "Bitrate"),
            ("size", "Размер файла"),
        ]
        self.value_labels: dict[str, QLabel] = {}
        for key, title in fields:
            label = QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(label.textInteractionFlags())
            self.value_labels[key] = label
            layout.addRow(f"{title}:", label)

    def clear(self) -> None:
        for label in self.value_labels.values():
            label.setText("—")

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        duration = metadata.get("duration")
        sample_rate = metadata.get("sample_rate")
        channels = metadata.get("channels")
        bitrate = metadata.get("bitrate")

        values = {
            "name": metadata.get("name") or "Недоступно",
            "path": metadata.get("path") or "Недоступно",
            "duration": f"{duration:.2f} с" if duration is not None else "Недоступно",
            "sample_rate": f"{sample_rate} Гц" if sample_rate else "Недоступно",
            "channels": str(channels) if channels else "Недоступно",
            "format": metadata.get("format") or "Недоступно",
            "subtype": metadata.get("subtype") or "Недоступно",
            "bitrate": f"{bitrate / 1000:.1f} кбит/с" if bitrate else "Недоступно",
            "size": _format_size(metadata.get("size_bytes")),
        }
        for key, value in values.items():
            self.value_labels[key].setText(value)

