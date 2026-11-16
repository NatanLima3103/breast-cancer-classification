"""Métricas, relatórios e gráficos de avaliação dos modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (execução em linha de comando)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

from src import config  # noqa: E402

LABELS: list[str] = list(config.CLASS_LABELS)


def cv_scoring() -> dict[str, object]:
    """Métricas calculadas em cada *fold* da validação cruzada.

    As métricas binárias usam ``recurrence-events`` como classe positiva.
    """
    pos = config.POSITIVE_CLASS
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, pos_label=pos, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=pos, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=pos, zero_division=0),
    }


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Accuracy, balanced accuracy, precision, recall e F1 (classe positiva)."""
    pos = config.POSITIVE_CLASS
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)),
    }


def compute_confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    """Matriz de confusão com linhas = classe real e colunas = classe prevista."""
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def full_classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> str:
    """Relatório de classificação completo (por classe, macro e ponderado)."""
    return classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0)


def format_confusion_matrix(cm: np.ndarray) -> str:
    """Representação textual da matriz de confusão."""
    frame = pd.DataFrame(
        cm,
        index=[f"real: {label}" for label in LABELS],
        columns=[f"prev: {label}" for label in LABELS],
    )
    return frame.to_string()


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(config.COLOR_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(config.COLOR_GRID)
    ax.tick_params(colors=config.COLOR_TEXT_SECONDARY, labelsize=9)
    ax.title.set_color(config.COLOR_TEXT)
    ax.xaxis.label.set_color(config.COLOR_TEXT_SECONDARY)
    ax.yaxis.label.set_color(config.COLOR_TEXT_SECONDARY)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    """Salva a matriz de confusão como imagem (contagens e % por classe real)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=config.COLOR_SURFACE)
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if row_pct[i, j] > 0.6 else config.COLOR_TEXT
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1%})", ha="center", va="center",
                    color=color, fontsize=12)
    ax.set_xticks(range(len(LABELS)), labels=LABELS)
    ax.set_yticks(range(len(LABELS)), labels=LABELS, rotation=90, va="center")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(title, fontsize=11, loc="left")
    _style_axes(ax)
    fig.text(0.01, 0.01, "Percentuais calculados por linha (classe real).",
             fontsize=8, color=config.COLOR_TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Gráfico de barras horizontais: acurácia média (± desvio) na validação cruzada de
    cada algoritmo e, apenas para o vencedor e a referência majoritária, a acurácia no teste.
    """
    df = comparison.sort_values("cv_accuracy_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    height = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(df) + 1.6), facecolor=config.COLOR_SURFACE)

    is_baseline = df["is_baseline"].to_numpy()
    cv_colors = [config.COLOR_BASELINE if b else config.COLOR_NEGATIVE for b in is_baseline]
    test_colors = [config.COLOR_BASELINE if b else config.COLOR_POSITIVE for b in is_baseline]
    ax.barh(y + height / 2, df["cv_accuracy_mean"], height=height - 0.04,
            xerr=df["cv_accuracy_std"], color=cv_colors, edgecolor=config.COLOR_SURFACE,
            error_kw={"ecolor": config.COLOR_TEXT_SECONDARY, "elinewidth": 1, "capsize": 3},
            label="Validação cruzada (média ± desvio-padrão)")
    # Apenas o modelo vencedor e a referência são avaliados no teste; os demais (NaN)
    # não recebem barra de teste.
    has_test = df["test_accuracy"].notna().to_numpy()
    ax.barh(y[has_test] - height / 2, df.loc[has_test, "test_accuracy"], height=height - 0.04,
            color=[c for c, h in zip(test_colors, has_test) if h],
            edgecolor=config.COLOR_SURFACE, alpha=0.85)
    for yi, (cv_mean, cv_std, test_acc) in enumerate(
        zip(df["cv_accuracy_mean"], df["cv_accuracy_std"], df["test_accuracy"])
    ):
        ax.text(cv_mean + cv_std + 0.008, yi + height / 2, f"{cv_mean:.3f}", va="center",
                fontsize=8, color=config.COLOR_TEXT)
        test_label = f"{test_acc:.3f}" if pd.notna(test_acc) else "teste não avaliado"
        test_x = test_acc + 0.012 if pd.notna(test_acc) else 0.505
        ax.text(test_x, yi - height / 2, test_label, va="center",
                fontsize=8, color=config.COLOR_TEXT_SECONDARY)

    ax.set_yticks(y, labels=df["model"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Acurácia")
    ax.grid(axis="x", color=config.COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Comparação de acurácia dos algoritmos (seleção apenas pela validação cruzada)\n"
                 "Teste avaliado somente para o modelo vencedor e a referência, após a seleção",
                 fontsize=11, loc="left")
    handles = [
        Patch(color=config.COLOR_NEGATIVE, label="Validação cruzada (média ± desvio-padrão)"),
        Patch(color=config.COLOR_POSITIVE, alpha=0.85, label="Conjunto de teste (vencedor)"),
        Patch(color=config.COLOR_BASELINE, label="Referência: sempre a classe majoritária"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
