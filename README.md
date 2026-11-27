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
- [Сравнение датасетов](docs/DATASET_COMPARISON.md)
- [Воспроизводимый notebook исследования](audio_deepfake_detector/notebooks/dataset_research.ipynb)
- [Подготовка RuASD и обучение CNN в Colab](docs/RUASD_COLAB.md)
- [Обучение RuASD из локального кэша Colab](audio_deepfake_detector/notebooks/03_ruasd_train_cnn.ipynb)
- [Проверка и продолжение сборки TAR на CPU](audio_deepfake_detector/notebooks/04_ruasd_tar_archives_cpu.ipynb)
- [Восстановление готовых TAR и обучение CNN](audio_deepfake_detector/notebooks/05_ruasd_train_simple.ipynb)
- [Независимая оценка CNN на test_unseen](audio_deepfake_detector/notebooks/06_ruasd_evaluate_cnn.ipynb)
- [Внешняя оценка CNN на своих REAL/FAKE записях](audio_deepfake_detector/notebooks/07_external_audio_evaluation.ipynb)
- [Диагностика обобщения CNN](audio_deepfake_detector/notebooks/08_diagnose_cnn_generalization.ipynb)
- [Журнал экспериментов ВКР](docs/VKR_EXPERIMENT_LOG.md) · [Отчёт Word](docs/VKR_EXPERIMENT_REPORT.docx)

Тема переключается кнопкой «Тема» рядом с кнопкой анализа и сохраняется между
запусками. Локальные окна «Руководство пользователя» и «О программе» доступны
в меню «Справка».

CNN пока не обучена: приложение честно показывает статус
`МОДЕЛЬ НЕ ЗАГРУЖЕНА` и не генерирует случайные REAL/FAKE-результаты.

