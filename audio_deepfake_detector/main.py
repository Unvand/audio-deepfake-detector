"""Совместимая точка запуска из каталога проекта."""

import sys
from pathlib import Path


def main() -> int:
    """Запустить desktop-приложение."""
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audio_deepfake_detector.app.application import run_application

    return run_application()


if __name__ == "__main__":
    raise SystemExit(main())
