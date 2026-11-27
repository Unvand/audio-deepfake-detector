"""Пересобрать обезличенные графики из сохранённых результатов Drive."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'experiment_results'


def main():
    history = json.loads((ROOT / 'training_history.json').read_text())
    epochs = [r['epoch'] + 1 for r in history]
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), layout='constrained')
    for key in ('train_loss', 'validation_loss'):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    axes[0].legend()
    axes[0].set_ylabel('Потери')
    for ax, key in zip(axes[1:3], ('f1', 'eer'), strict=True):
        ax.plot(epochs, [r['validation_source'][key] for r in history], 'o-')
        ax.set_ylabel('Validation ' + key.upper())
    axes[3].plot(epochs, [r['learning_rate'] for r in history])
    axes[3].set(xlabel='Эпоха', ylabel='Learning rate')
    fig.savefig(ROOT / 'training.png', dpi=160)
    plt.close(fig)
    frame = pd.read_csv(ROOT / 'external_anonymous.csv')
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    frame.pivot(index='speaker', columns='true_class', values='fake_score').plot.bar(ax=ax)
    ax.axhline(.5, linestyle='--', color='black')
    ax.set(xlabel='Обезличенный диктор', ylabel='FAKE-score', title='Внешний тест: исходные записи')
    ax.tick_params(axis='x', labelrotation=0)
    fig.savefig(ROOT / 'external.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
