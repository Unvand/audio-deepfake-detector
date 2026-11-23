# Audio Deepfake Detector

Учебный проект ВКР по обнаружению синтезированной речи. Основной обучающий
pipeline — одна CNN на log-Mel-спектрограммах RuASD. Репозиторий содержит
запускаемый desktop-прототип на PyQt6, но обученная модель в GUI пока не
подключена и приложение не выдаёт фиктивные REAL/FAKE.

## Текущий этап

- подготовлена структура проекта;
- добавлен Colab notebook для первичного анализа официального протокола
  ASVspoof 5 без скачивания аудиодатасета;
- добавлены сравнительное исследование современных датасетов и безопасный
  notebook, который получает небольшие metadata/protocol без автоматической
  загрузки аудиокорпусов;
- добавлены два независимых RuASD notebook: CPU-подготовка FLAC с возобновлением
  и управляемое через `ipywidgets` GPU-обучение одной CNN с TensorBoard и
  checkpoint после каждой эпохи;
- добавлен отдельный notebook 03 с проверяемым локальным кэшем `/content` и
  диагностикой времени Drive/FLAC/Mel/GPU/CNN без изменения baseline;
- notebook 04 проверяет и продолжает упаковку TAR в Colab без ускорителя;
- кнопка «Тема» переключает светлую и тёмную палитры и сохраняет выбор.
  Руководство и сведения о версии доступны в меню «Справка»;
- реализованы загрузка аудио, stereo-to-mono, resampling до 16 кГц и разбиение
  записи на окна заданной длительности;
- добавлен GUI с waveform, обычной спектрограммой, F0, MFCC и спектральными
  характеристиками;
- тяжёлый анализ выполняется в отдельном `QThread`, интерфейс показывает этапы,
  прогресс и понятные сообщения об ошибках;
- подготовлен отдельный воспроизводимый Colab-pipeline обучения; до появления и
  явного подключения реального checkpoint статус GUI остаётся
  `МОДЕЛЬ НЕ ЗАГРУЖЕНА`.

## Запуск desktop-приложения

Рекомендуется Python 3.10 или новее.

```bash
cd audio_deepfake_detector
pip install -r requirements.txt
python main.py
```

Рекомендуемый запуск из корня клонированного репозитория:

```bash
pip install -r requirements.txt
python main.py
```

Поддерживаемые форматы выбора: WAV, FLAC, MP3 и OGG. Возможность декодирования
конкретного файла зависит от codec-поддержки установленной библиотеки.

## Структура

```text
audio_deepfake_detector/
├── main.py
├── app/
│   ├── application.py
│   ├── main_window.py
│   ├── help_dialogs.py
│   ├── workers.py
│   ├── styles.py
│   └── widgets/
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── dataset_research.ipynb
│   ├── 01_ruasd_prepare_audio.ipynb
│   ├── 02_ruasd_train_cnn.ipynb
│   ├── 03_ruasd_train_cnn.ipynb
│   └── 04_ruasd_tar_archives_cpu.ipynb
├── research/
│   └── dataset_metadata.py
├── src/
│   ├── __init__.py
│   ├── audio_analysis.py
│   ├── preprocessing.py
│   ├── ruasd_preparation.py
│   ├── cnn_training.py
│   ├── training_metrics.py
│   ├── colab_cache.py
│   ├── colab_tar_cache.py
│   ├── training_benchmark.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── models/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Установка в Google Colab

```python
!pip install -r requirements-research.txt
```

## Данные

Датасеты, аудиофайлы, локальные протоколы и веса моделей не должны попадать в
Git. `dataset_research.ipynb` умеет получить официальный небольшой protocol
ASVspoof 5; перед любой большой загрузкой он показывает объём и требует явное
подтверждение. Полное сравнение и ссылки находятся в
[`docs/DATASET_COMPARISON.md`](../docs/DATASET_COMPARISON.md).

Зависимости notebook вынесены в `requirements-research.txt`, чтобы pandas и
seaborn не увеличивали минимальную desktop-установку.

RuASD pipeline использует отдельные лёгкие наборы Colab-зависимостей для
подготовки и обучения. Пошаговый запуск, структура результатов и восстановление
описаны в [`docs/RUASD_COLAB.md`](../docs/RUASD_COLAB.md).

Собственный тестовый набор с настоящими и RVC/MuXVS-записями предназначен
только для финального внешнего тестирования и не должен использоваться при
обучении.

## Важное ограничение прототипа

F0, MFCC и спектральные характеристики являются только описательным анализом.
Они не применяют пороги и не формируют вердикт REAL/FAKE. Класс
`AudioDeepfakeInference` в `src/inference.py` служит точкой подключения будущей
CNN; до загрузки обученной модели он явно сообщает, что анализ не выполнен.

