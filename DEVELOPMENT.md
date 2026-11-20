# Разработка

## Требования

- Python 3.10 или новее;
- Windows 10/11, Linux или macOS;
- Git.

## Создание виртуального окружения

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Если PowerShell запрещает запуск локальных скриптов, можно активировать
окружение через `cmd.exe`:

```bat
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Установка

Только desktop-приложение:

```bash
pip install -r requirements.txt
```

Приложение и тесты:

```bash
pip install -r requirements-dev.txt
```

Для работы с исследовательским notebook дополнительно нужны pandas и seaborn:

```bash
pip install -r requirements-research.txt
```

## Запуск

Из корня репозитория:

```bash
python main.py
```

## Тесты

```bash
python -m pytest -q
```

Проверка импортов и базового качества кода:

```bash
python -m ruff check .
```

GUI-тесты автоматически используют headless-платформу Qt. Они проверяют, что
вычисления идут не в главном потоке и что закрытие окна запрашивает безопасную
отмену worker перед завершением `QThread`.

## Правила работы с данными

- не добавлять в Git аудиофайлы, датасеты и архивы ASVspoof;
- не добавлять веса, checkpoints и локальные результаты;
- собственные RVC/MuXVS-записи использовать только для отдельного тестирования;
- не формировать REAL/FAKE без реально загруженной модели;
- параметры Mel-спектрограммы и архитектуру CNN выбирать отдельным
  исследовательским этапом.

