"""Корневая точка запуска после клонирования репозитория."""

from audio_deepfake_detector.app.application import run_application


def main() -> int:
    """Запустить desktop-приложение."""
    return run_application()


if __name__ == "__main__":
    raise SystemExit(main())

