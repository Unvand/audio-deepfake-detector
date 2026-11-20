# Анализ синтезированной речи

Запускаемый PyQt6-прототип для ВКР по обнаружению синтезированной речи.

```bash
pip install -r requirements.txt
python main.py
```

Основной код, notebook и подробное описание находятся в каталоге
[`audio_deepfake_detector`](audio_deepfake_detector/README.md).

- [Архитектура](ARCHITECTURE.md)
- [Инструкция разработчика](DEVELOPMENT.md)

CNN пока не обучена: приложение честно показывает статус
`МОДЕЛЬ НЕ ЗАГРУЖЕНА` и не генерирует случайные REAL/FAKE-результаты.

