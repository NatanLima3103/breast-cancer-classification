"""Módulo de treinamento: compara algoritmos, seleciona e persiste o melhor modelo.

Uso:
    python -m src.training
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import load_dataset, split_features_target
from src.evaluation import (
    compute_confusion_matrix,
    compute_metrics,
    cv_scoring,
    format_confusion_matrix,
    full_classification_report,
    plot_accuracy_comparison,
    plot_confusion_matrix,
)
from src.preprocessing import CLASSIFIER_STEP, NO_SAMPLER, SAMPLER_STEP, build_pipeline

logger = logging.getLogger(__name__)

BASELINE_NAME = "Baseline (classe majoritária)"


@dataclass(frozen=True)
class ModelSpec:
    """Definição de um algoritmo candidato e do seu espaço de hiperparâmetros."""

    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list] | list[dict[str, list]]
    scale: bool
    search: str = "grid"  # "grid" (GridSearchCV) ou "random" (RandomizedSearchCV)
    n_iter: int | None = None
    is_baseline: bool = False


@dataclass
class TrainingResult:
    """Resultado consolidado de uma execução de treinamento."""

    comparison: pd.DataFrame
    best_model_name: str
    best_pipeline: Pipeline
    selection_rationale: str
    test_metrics: dict[str, float]
    confusion: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)
    smote_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)


def _prefixed(grid: dict[str, list], random_state: int = config.RANDOM_STATE) -> dict[str, list]:
    """Prefixa os hiperparâmetros do classificador e acrescenta a opção de balanceamento.

    ``sampler`` é avaliado como hiperparâmetro: sem balanceamento ("passthrough") ou
    com SMOTE. O imblearn aplica o SMOTE apenas no treino de cada *fold*.
    """
    prefixed: dict[str, list] = {
        f"{CLASSIFIER_STEP}__{key}": values for key, values in grid.items()
    }
    prefixed[SAMPLER_STEP] = [NO_SAMPLER, SMOTE(random_state=random_state)]
    return prefixed


def get_model_specs(random_state: int = config.RANDOM_STATE) -> list[ModelSpec]:
    """Algoritmos comparados e grades de hiperparâmetros (modestas, base pequena)."""
    return [
        ModelSpec(
            name=BASELINE_NAME,
            estimator=DummyClassifier(strategy="most_frequent"),
            param_grid={},
            scale=False,
            is_baseline=True,
        ),
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(max_iter=5000, random_state=random_state),
            param_grid=_prefixed({
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            }, random_state),
            scale=True,
        ),
        ModelSpec(
            name="Decision Tree",
            estimator=DecisionTreeClassifier(random_state=random_state),
            param_grid=_prefixed({
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 3, 4, 5, None],
                "min_samples_leaf": [1, 5, 10],
            }, random_state),
            scale=False,
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(random_state=random_state, n_jobs=1),
            param_grid=_prefixed({
                "n_estimators": [100, 300],
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", "log2", 0.5],
                "class_weight": [None, "balanced_subsample"],
            }, random_state),
            scale=False,
            search="random",
            n_iter=30,  # espaço maior com a opção de SMOTE
        ),
        ModelSpec(
            name="SVC",
            estimator=SVC(random_state=random_state),
            param_grid=[
                _prefixed({
                    "kernel": ["linear"],
                    "C": [0.01, 0.1, 1.0],
                    "class_weight": [None, "balanced"],
                }, random_state),
                _prefixed({
                    "kernel": ["rbf"],
                    "C": [0.1, 1.0, 10.0],
                    "gamma": ["scale", 0.01, 0.1],
                    "class_weight": [None, "balanced"],
                }, random_state),
            ],
            scale=True,
        ),
        ModelSpec(
            name="KNN",
            estimator=KNeighborsClassifier(),
            param_grid=_prefixed({
                "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            }, random_state),
            scale=True,
        ),
        ModelSpec(
            name="Gaussian Naive Bayes",
            estimator=GaussianNB(),
            param_grid=_prefixed({"var_smoothing": list(np.logspace(-9, 0, 10))}, random_state),
            scale=False,
        ),
    ]


def make_cv(
    n_splits: int = config.CV_N_SPLITS,
    n_repeats: int = config.CV_N_REPEATS,
    random_state: int = config.RANDOM_STATE,
) -> RepeatedStratifiedKFold:
    """Validação cruzada estratificada repetida (5 divisões × N repetições)."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divisão treino/teste estratificada (80/20) com semente fixa."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def build_search(
    spec: ModelSpec, cv: BaseCrossValidator, n_jobs: int = config.N_JOBS
) -> GridSearchCV | RandomizedSearchCV:
    """Cria a busca de hiperparâmetros sobre o pipeline completo (pré-processamento
    + balanceamento opcional + classificador), de modo que o pré-processamento e o
    SMOTE sejam ajustados somente com os dados de treino de cada *fold*.
    """
    pipeline = build_pipeline(spec.estimator, scale=spec.scale)
    common: dict[str, Any] = {
        "scoring": cv_scoring(),
        "refit": "accuracy",  # reajusta o melhor candidato com todo o conjunto de treino
        "cv": cv,
        "n_jobs": n_jobs,
        "return_train_score": False,
        "error_score": "raise",  # erros de ajuste não são silenciados
    }
    if spec.search == "random":
        return RandomizedSearchCV(
            pipeline,
            param_distributions=spec.param_grid,
            n_iter=spec.n_iter or 10,
            random_state=config.RANDOM_STATE,
            **common,
        )
    return GridSearchCV(pipeline, param_grid=spec.param_grid, **common)


def _cv_row(search: GridSearchCV | RandomizedSearchCV) -> dict[str, float]:
    """Extrai as métricas de validação cruzada do melhor candidato da busca."""
    res = search.cv_results_
    i = search.best_index_
    row: dict[str, float] = {}
    for metric in cv_scoring():
        row[f"cv_{metric}_mean"] = float(res[f"mean_test_{metric}"][i])
        row[f"cv_{metric}_std"] = float(res[f"std_test_{metric}"][i])
    return row


def smote_comparison_rows(
    model_name: str, search: GridSearchCV | RandomizedSearchCV
) -> list[dict[str, Any]]:
    """Compara, para um algoritmo, o melhor candidato sem SMOTE e o melhor com SMOTE.

    Os números vêm de ``cv_results_`` (validação cruzada no treino). Em cada grupo é
    escolhido o candidato de melhor posição em acurácia (mesma regra do ``refit``).
    """
    res = search.cv_results_
    if f"param_{SAMPLER_STEP}" not in res:
        return []
    samplers = list(res[f"param_{SAMPLER_STEP}"])
    rows: list[dict[str, Any]] = []
    for uses_smote in (False, True):
        idx = [i for i, v in enumerate(samplers) if isinstance(v, SMOTE) == uses_smote]
        row: dict[str, Any] = {"model": model_name, "smote": uses_smote, "n_candidates": len(idx)}
        if idx:
            best = min(idx, key=lambda i: (res["rank_test_accuracy"][i], i))
            params = {k: v for k, v in res["params"][best].items() if k != SAMPLER_STEP}
            row.update({
                "cv_accuracy_mean": float(res["mean_test_accuracy"][best]),
                "cv_accuracy_std": float(res["std_test_accuracy"][best]),
                "cv_recall_mean": float(res["mean_test_recall"][best]),
                "cv_recall_std": float(res["std_test_recall"][best]),
                "best_params": json.dumps(_clean_params(params), default=str),
                "chosen_by_search": bool(best == search.best_index_),
            })
        else:  # possível apenas no RandomizedSearchCV, se nenhum sorteio cair no grupo
            row.update({
                "cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan,
                "cv_recall_mean": np.nan, "cv_recall_std": np.nan,
                "best_params": "", "chosen_by_search": False,
            })
        rows.append(row)
    return rows


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove o prefixo do passo e converte tipos NumPy para tipos nativos."""
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            value = value.item()
        clean[key.replace(f"{CLASSIFIER_STEP}__", "")] = value
    return clean


def select_best_model(
    comparison: pd.DataFrame, tolerance: float = config.ACCURACY_TIE_TOLERANCE
) -> tuple[str, str]:
    """Seleciona o melhor modelo usando apenas métricas de validação cruzada.

    Critério principal: maior acurácia média na validação cruzada.
    Critério secundário: entre os modelos cuja acurácia média está a até
    ``tolerance`` da melhor, escolhe o de maior recall médio da classe
    ``recurrence-events``. A referência majoritária nunca é selecionável.
    """
    candidates = comparison.loc[~comparison["is_baseline"]].copy()
    if candidates.empty:
        raise ValueError("Nenhum modelo candidato disponível para seleção.")
    best_acc = candidates["cv_accuracy_mean"].max()
    leader = candidates.loc[candidates["cv_accuracy_mean"].idxmax(), "model"]
    tied = candidates.loc[candidates["cv_accuracy_mean"] >= best_acc - tolerance]
    tied = tied.sort_values(["cv_recall_mean", "cv_accuracy_mean"], ascending=[False, False])
    winner = str(tied.iloc[0]["model"])

    if len(tied) == 1:
        rationale = (
            f"'{winner}' obteve a maior acurácia média na validação cruzada "
            f"({best_acc:.4f}) e nenhum outro modelo ficou a até {tolerance:.3f} dela; "
            "o critério secundário (recall de recurrence-events) não foi necessário."
        )
    else:
        listing = "; ".join(
            f"{r.model}: acurácia={r.cv_accuracy_mean:.4f}, recall={r.cv_recall_mean:.4f}"
            for r in tied.itertuples()
        )
        if winner == leader:
            outcome = f"'{winner}' tem a maior acurácia e também o maior recall entre eles"
        else:
            outcome = (
                f"'{winner}' foi escolhido por ter o maior recall médio de recurrence-events, "
                f"embora '{leader}' tenha a maior acurácia média ({best_acc:.4f})"
            )
        rationale = (
            f"{len(tied)} modelos ficaram a até {tolerance:.3f} da melhor acurácia média "
            f"de validação cruzada ({best_acc:.4f}) e foram considerados empatados "
            f"[{listing}]. Aplicado o critério secundário: {outcome}."
        )
    return winner, rationale


def run_training(
    data_path: Path | str = config.DATA_PATH,
    models_dir: Path | str = config.MODELS_DIR,
    reports_dir: Path | str = config.REPORTS_DIR,
    specs: list[ModelSpec] | None = None,
    cv: BaseCrossValidator | None = None,
    n_jobs: int = config.N_JOBS,
) -> TrainingResult:
    """Executa o fluxo completo de treinamento, seleção, avaliação e persistência."""
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    specs = specs if specs is not None else get_model_specs()
    cv = cv if cv is not None else make_cv()

    df = load_dataset(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    logger.info("Treino: %d registros | Teste: %d registros", len(X_train), len(X_test))

    rows: list[dict[str, Any]] = []
    smote_rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for spec in specs:
        search = build_search(spec, cv, n_jobs=n_jobs)
        n_candidates = (
            spec.n_iter if spec.search == "random" else len(ParameterGrid(spec.param_grid))
        )
        logger.info("Ajustando %s (%d configurações)...", spec.name, n_candidates)
        search.fit(X_train, y_train)  # somente dados de treinamento
        best: Pipeline = search.best_estimator_
        fitted[spec.name] = best
        smote_rows.extend(smote_comparison_rows(spec.name, search))
        # Apenas diagnóstico (ressubstituição); o conjunto de teste NÃO é consultado aqui.
        train_metrics = compute_metrics(y_train, best.predict(X_train))
        row: dict[str, Any] = {
            "model": spec.name,
            "is_baseline": spec.is_baseline,
            "search": "RandomizedSearchCV" if spec.search == "random" else "GridSearchCV",
            "n_candidates": n_candidates,
            "best_params": json.dumps(_clean_params(search.best_params_), default=str),
            **_cv_row(search),
            "train_accuracy": train_metrics["accuracy"],
        }
        rows.append(row)
        logger.info(
            "  CV acurácia=%.4f ± %.4f | CV recall(recurrence)=%.4f",
            row["cv_accuracy_mean"], row["cv_accuracy_std"], row["cv_recall_mean"],
        )

    comparison = pd.DataFrame(rows)
    smote_comparison = pd.DataFrame(smote_rows)
    winner, rationale = select_best_model(comparison)  # somente métricas de CV
    comparison["selected"] = comparison["model"] == winner
    logger.info("Modelo selecionado: %s", winner)

    # Somente após a seleção o conjunto de teste é usado, e apenas para o vencedor e
    # para a referência majoritária (DummyClassifier) definida previamente. Os demais
    # candidatos ficam com as colunas test_* vazias (NaN).
    # O GridSearchCV/RandomizedSearchCV (refit="accuracy") já reajustou o melhor
    # pipeline com todo o conjunto de treinamento; o teste não participou do ajuste.
    best_pipeline = fitted[winner]
    y_pred = best_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)
    evaluated: dict[str, dict[str, float]] = {winner: test_metrics}
    baseline_names = comparison.loc[comparison["is_baseline"], "model"].tolist()
    baseline_cm: np.ndarray | None = None
    for name in baseline_names:
        baseline_pred = fitted[name].predict(X_test)
        evaluated[name] = compute_metrics(y_test, baseline_pred)
        baseline_cm = compute_confusion_matrix(y_test, baseline_pred)

    metric_names = list(test_metrics)
    for metric in metric_names:
        comparison[f"test_{metric}"] = comparison["model"].map(
            lambda m, k=metric: evaluated[m][k] if m in evaluated else np.nan
        )
    comparison["evaluated_on_test"] = comparison["model"].isin(evaluated)
    comparison = comparison.sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
    baseline_row = comparison.loc[comparison["is_baseline"]]
    logger.info(
        "Teste (somente vencedor e referência): %s",
        " | ".join(f"{m}: acurácia={v['accuracy']:.4f}" for m, v in evaluated.items()),
    )
    baseline_metrics = evaluated[baseline_names[0]] if baseline_names else None

    model_path = models_dir / config.MODEL_PATH.name
    joblib.dump(best_pipeline, model_path)

    summary = _build_summary(
        comparison, winner, rationale, best_pipeline, test_metrics, cm, baseline_row,
        baseline_metrics, baseline_cm, X_train, X_test, y_train, y_test, cv, model_path,
    )
    _write_reports(
        reports_dir, comparison, winner, rationale, summary, y_test, y_pred, cm, smote_comparison
    )
    return TrainingResult(
        comparison=comparison,
        best_model_name=winner,
        best_pipeline=best_pipeline,
        selection_rationale=rationale,
        test_metrics=test_metrics,
        confusion=cm,
        summary=summary,
        smote_comparison=smote_comparison,
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _build_summary(
    comparison: pd.DataFrame,
    winner: str,
    rationale: str,
    pipeline: Pipeline,
    test_metrics: dict[str, float],
    cm: np.ndarray,
    baseline_row: pd.DataFrame,
    baseline_metrics: dict[str, float] | None,
    baseline_cm: np.ndarray | None,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: BaseCrossValidator,
    model_path: Path,
) -> dict[str, Any]:
    win = comparison.loc[comparison["model"] == winner].iloc[0]
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "cross_validation": {
            "strategy": type(cv).__name__,
            "n_splits": getattr(cv, "cvargs", {}).get("n_splits", None),
            "n_repeats": getattr(cv, "n_repeats", None),
            "n_folds_total": cv.get_n_splits(),
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_class_distribution": {k: int(v) for k, v in y_train.value_counts().items()},
        "test_class_distribution": {k: int(v) for k, v in y_test.value_counts().items()},
        "positive_class": config.POSITIVE_CLASS,
        "selection_criterion": (
            "Maior acurácia média na validação cruzada (somente dados de treino); "
            f"modelos a até {config.ACCURACY_TIE_TOLERANCE} desta acurácia são considerados "
            "empatados e o desempate é o maior recall médio de recurrence-events."
        ),
        "selected_model": winner,
        "selection_rationale": rationale,
        "best_params": json.loads(win["best_params"]),
        "classifier": repr(pipeline.named_steps[CLASSIFIER_STEP]),
        "cv_metrics": {
            k.removeprefix("cv_"): float(win[k]) for k in win.index if str(k).startswith("cv_")
        },
        "train_accuracy_resubstitution": float(win["train_accuracy"]),
        "test_set_usage": (
            "O conjunto de teste foi usado somente após a seleção, e apenas para avaliar o "
            "modelo vencedor e a referência majoritária (DummyClassifier) definida previamente. "
            "Os demais candidatos não foram avaliados no teste (colunas test_* vazias)."
        ),
        "models_evaluated_on_test": comparison.loc[
            comparison["evaluated_on_test"], "model"
        ].tolist(),
        "test_metrics": test_metrics,
        "test_confusion_matrix": {
            "labels": list(config.CLASS_LABELS),
            "matrix": cm.tolist(),
        },
        "baseline": {
            "model": str(baseline_row["model"].iloc[0]),
            "cv_accuracy_mean": float(baseline_row["cv_accuracy_mean"].iloc[0]),
            "cv_accuracy_std": float(baseline_row["cv_accuracy_std"].iloc[0]),
            "test_metrics": baseline_metrics,
            "test_confusion_matrix": baseline_cm.tolist() if baseline_cm is not None else None,
        } if not baseline_row.empty else None,
        "model_path": _relative(model_path),
        "features": list(config.FEATURES),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    return summary


def _write_reports(
    reports_dir: Path,
    comparison: pd.DataFrame,
    winner: str,
    rationale: str,
    summary: dict[str, Any],
    y_test: pd.Series,
    y_pred: np.ndarray,
    cm: np.ndarray,
    smote_comparison: pd.DataFrame,
) -> None:
    comparison.to_csv(reports_dir / config.COMPARISON_CSV_FILE, index=False, float_format="%.4f")
    smote_comparison.to_csv(
        reports_dir / config.SMOTE_COMPARISON_FILE, index=False, float_format="%.4f"
    )
    with open(reports_dir / config.TRAINING_SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    display_cols = [
        "model", "cv_accuracy_mean", "cv_accuracy_std", "cv_recall_mean",
        "cv_f1_mean", "test_accuracy", "test_precision", "test_recall", "test_f1",
    ]
    table = comparison[display_cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"
    )
    m = summary["test_metrics"]
    cvm = summary["cv_metrics"]
    lines = [
        "RELATÓRIO DE CLASSIFICAÇÃO — MODELO SELECIONADO",
        "=" * 70,
        f"Modelo selecionado: {winner}",
        f"Hiperparâmetros: {json.dumps(summary['best_params'], ensure_ascii=False)}",
        f"Semente aleatória: {summary['random_state']}",
        f"Validação: {summary['cross_validation']['strategy']} "
        f"({summary['cross_validation']['n_folds_total']} ajustes por configuração)",
        "",
        "Justificativa da seleção (somente validação cruzada no treino):",
        f"  {rationale}",
        "",
        "Validação cruzada (treino):",
        f"  Acurácia média: {cvm['accuracy_mean']:.4f} (desvio-padrão {cvm['accuracy_std']:.4f})",
        f"  Recall recurrence-events médio: {cvm['recall_mean']:.4f} "
        f"(desvio-padrão {cvm['recall_std']:.4f})",
        f"  Precision média: {cvm['precision_mean']:.4f} | F1 médio: {cvm['f1_mean']:.4f}",
        "",
        f"Conjunto de teste ({summary['n_test']} registros), usado somente após a seleção:",
        f"  Accuracy:          {m['accuracy']:.4f}",
        f"  Balanced accuracy: {m['balanced_accuracy']:.4f}",
        f"  Precision (recurrence-events): {m['precision']:.4f}",
        f"  Recall    (recurrence-events): {m['recall']:.4f}",
        f"  F1-score  (recurrence-events): {m['f1']:.4f}",
        "",
        "Relatório de classificação completo (teste):",
        full_classification_report(y_test, y_pred),
        "Matriz de confusão (teste):",
        format_confusion_matrix(cm),
        "",
    ]
    baseline = summary.get("baseline")
    if baseline is not None and baseline["test_metrics"] is not None:
        b_cv = baseline["cv_accuracy_mean"]
        bm = baseline["test_metrics"]
        lines += [
            "Referência previamente definida (DummyClassifier, sempre prevê "
            "no-recurrence-events; nunca selecionável):",
            f"  Acurácia CV: {b_cv:.4f} (desvio-padrão {baseline['cv_accuracy_std']:.4f})",
            f"  Teste: accuracy={bm['accuracy']:.4f} | balanced accuracy="
            f"{bm['balanced_accuracy']:.4f} | precision={bm['precision']:.4f} | "
            f"recall={bm['recall']:.4f} | f1={bm['f1']:.4f}",
            "  Matriz de confusão da referência (teste):",
            format_confusion_matrix(np.array(baseline["test_confusion_matrix"])),
            f"  Ganho do modelo selecionado sobre a referência: "
            f"CV {cvm['accuracy_mean'] - b_cv:+.4f} | teste {m['accuracy'] - bm['accuracy']:+.4f}",
            "",
        ]
        if cvm["accuracy_mean"] <= b_cv:
            lines.append(
                "ATENÇÃO: o modelo selecionado não supera a referência majoritária na "
                "validação cruzada."
            )
    lines += [
        "Comparação dos algoritmos (ordenada pela acurácia média de validação cruzada):",
        table,
        "",
    ]
    if not smote_comparison.empty:
        smote_view = smote_comparison.assign(
            balanceamento=smote_comparison["smote"].map({False: "sem SMOTE", True: "com SMOTE"})
        )[["model", "balanceamento", "cv_accuracy_mean", "cv_accuracy_std", "cv_recall_mean",
           "chosen_by_search"]]
        lines += [
            "Balanceamento: melhor candidato sem SMOTE x com SMOTE, por algoritmo",
            "(validação cruzada no treino; o SMOTE é aplicado só no treino de cada fold):",
            smote_view.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"),
            "",
        ]
    lines += [
        "Observações:",
        "  * A comparação e a seleção usam exclusivamente a validação cruzada no treino.",
        "  * O conjunto de teste foi usado somente após a seleção, e apenas para avaliar o",
        "    modelo vencedor e a referência majoritária definida previamente. Os demais",
        "    candidatos NÃO foram avaliados no teste: suas colunas test_* aparecem como '-'",
        "    (vazias no CSV).",
        "  * A acurácia isolada é enganosa com 70% de registros negativos: observe o recall",
        "    e o F1 de recurrence-events e a comparação com a referência majoritária.",
        f"  * Acurácia de treino por ressubstituição do modelo selecionado: "
        f"{summary['train_accuracy_resubstitution']:.4f} (não indica generalização).",
    ]
    (reports_dir / config.CLASSIFICATION_REPORT_FILE).write_text("\n".join(lines) + "\n",
                                                               encoding="utf-8")
    plot_confusion_matrix(
        cm, f"Matriz de confusão — {winner} (conjunto de teste)",
        reports_dir / config.CONFUSION_MATRIX_PLOT_FILE,
    )
    plot_accuracy_comparison(comparison, reports_dir / config.ACCURACY_PLOT_FILE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.training",
        description="Treina, compara e salva o melhor classificador para a base Breast Cancer.",
    )
    parser.add_argument("--data", type=Path, default=config.DATA_PATH, help="Caminho do CSV.")
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--n-repeats", type=int, default=config.CV_N_REPEATS,
                        help="Repetições da validação cruzada estratificada (5 divisões).")
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS,
                        help="Processos paralelos usados na busca de hiperparâmetros.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    result = run_training(
        data_path=args.data,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        cv=make_cv(n_repeats=args.n_repeats),
        n_jobs=args.n_jobs,
    )
    cols = ["model", "cv_accuracy_mean", "cv_accuracy_std", "cv_recall_mean", "test_accuracy",
            "test_recall"]
    print()
    print(result.comparison[cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"
    ))
    print()
    print(f"Modelo selecionado: {result.best_model_name}")
    print(f"Justificativa: {result.selection_rationale}")
    m = result.test_metrics
    print(f"Teste -> acurácia={m['accuracy']:.4f} | precision={m['precision']:.4f} | "
          f"recall={m['recall']:.4f} | f1={m['f1']:.4f}")
    print(f"Modelo salvo em: {Path(args.models_dir) / config.MODEL_PATH.name}")
    print(f"Relatórios salvos em: {args.reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
