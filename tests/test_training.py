"""Testes do treinamento, da seleção e da persistência do melhor modelo."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src import config
from src.data import load_dataset, split_features_target
from src.preprocessing import NO_SAMPLER, PREPROCESSOR_STEP, SAMPLER_STEP
from src.training import (
    BASELINE_NAME,
    build_search,
    get_model_specs,
    make_cv,
    select_best_model,
    split_train_test,
)


def test_required_algorithms_are_compared():
    names = {spec.name for spec in get_model_specs()}
    assert {
        "Logistic Regression", "Decision Tree", "Random Forest", "SVC", "KNN",
        "Gaussian Naive Bayes", BASELINE_NAME,
    } <= names


def test_train_test_split_is_stratified_disjoint_and_reproducible():
    X, y = split_features_target(load_dataset())
    X_tr, X_te, y_tr, y_te = split_train_test(X, y)
    assert len(X_te) == 58 and len(X_tr) == 228
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs((y_te == config.POSITIVE_CLASS).mean() - (y == config.POSITIVE_CLASS).mean()) < 0.01
    X_tr2, X_te2, _, _ = split_train_test(X, y)
    assert X_te.index.equals(X_te2.index) and X_tr.index.equals(X_tr2.index)


def test_cv_is_repeated_stratified_and_reproducible():
    X, y = split_features_target(load_dataset())
    cv = make_cv()
    assert cv.get_n_splits() == config.CV_N_SPLITS * config.CV_N_REPEATS
    first = [test.tolist() for _, test in make_cv().split(X, y)]
    second = [test.tolist() for _, test in make_cv().split(X, y)]
    assert first == second


def test_search_wraps_full_pipeline_to_avoid_leakage():
    """O pré-processamento deve estar dentro do estimador avaliado pela busca,
    para ser reajustado em cada fold da validação cruzada."""
    for spec in get_model_specs():
        search = build_search(spec, make_cv(n_repeats=1), n_jobs=1)
        assert isinstance(search.estimator, ImbPipeline)
        assert [n for n, _ in search.estimator.steps] == [
            PREPROCESSOR_STEP, SAMPLER_STEP, "classifier"
        ]
        assert search.refit == "accuracy"


def test_smote_is_a_hyperparameter_of_every_candidate_but_not_the_baseline():
    for spec in get_model_specs():
        grids = spec.param_grid if isinstance(spec.param_grid, list) else [spec.param_grid]
        if spec.is_baseline:
            assert all(SAMPLER_STEP not in g for g in grids)
            continue
        for grid in grids:
            options = grid[SAMPLER_STEP]
            assert options[0] == NO_SAMPLER
            assert isinstance(options[1], SMOTE) and options[1].random_state == config.RANDOM_STATE
    rf = next(s for s in get_model_specs() if s.name == "Random Forest")
    assert rf.search == "random" and rf.n_iter == 30


def test_smote_comparison_is_generated(trained_artifacts):
    path = trained_artifacts["reports_dir"] / config.SMOTE_COMPARISON_FILE
    assert path.is_file()
    smote = pd.read_csv(path)
    assert list(smote.columns) == [
        "model", "smote", "n_candidates", "cv_accuracy_mean", "cv_accuracy_std",
        "cv_recall_mean", "cv_recall_std", "best_params", "chosen_by_search",
    ]
    assert BASELINE_NAME not in set(smote["model"])  # a referência não tem sampler
    for model, group in smote.groupby("model"):
        assert sorted(group["smote"]) == [False, True], model
        assert group["chosen_by_search"].sum() == 1, model
        assert group[["cv_accuracy_mean", "cv_recall_mean"]].notna().all(axis=None)
    # O melhor de cada algoritmo na comparação SMOTE coincide com o model_comparison.
    comp = trained_artifacts["result"].comparison.set_index("model")
    for r in smote.loc[smote["chosen_by_search"]].itertuples():
        assert r.cv_accuracy_mean == pytest.approx(comp.loc[r.model, "cv_accuracy_mean"], abs=1e-4)
    report = (trained_artifacts["reports_dir"] / config.CLASSIFICATION_REPORT_FILE).read_text(
        encoding="utf-8")
    assert "sem SMOTE" in report and "com SMOTE" in report


def _comparison(rows):
    return pd.DataFrame(rows, columns=["model", "is_baseline", "cv_accuracy_mean",
                                       "cv_recall_mean"])


def test_selection_uses_highest_cv_accuracy():
    comp = _comparison([
        ("A", False, 0.75, 0.30),
        ("B", False, 0.72, 0.60),
        ("Base", True, 0.70, 0.00),
    ])
    winner, rationale = select_best_model(comp, tolerance=0.01)
    assert winner == "A"
    assert "não foi necessário" in rationale


def test_selection_tie_break_by_recall():
    comp = _comparison([
        ("A", False, 0.750, 0.30),
        ("B", False, 0.745, 0.45),
        ("C", False, 0.700, 0.90),
    ])
    winner, rationale = select_best_model(comp, tolerance=0.01)
    assert winner == "B"  # empatado com A e com maior recall; C está fora da tolerância
    assert "critério secundário" in rationale


def test_baseline_is_never_selected():
    comp = _comparison([("Base", True, 0.90, 0.0), ("A", False, 0.60, 0.5)])
    winner, _ = select_best_model(comp)
    assert winner == "A"


def test_training_produces_comparison_and_selects_non_baseline(trained_artifacts):
    result = trained_artifacts["result"]
    comp = result.comparison
    assert set(comp["model"]) == {BASELINE_NAME, "Logistic Regression", "Gaussian Naive Bayes"}
    assert comp["selected"].sum() == 1
    assert result.best_model_name != BASELINE_NAME
    baseline = comp.loc[comp["is_baseline"]].iloc[0]
    # A referência majoritária prevê sempre a classe negativa: recall zero.
    assert baseline["test_recall"] == 0.0
    assert baseline["test_accuracy"] == pytest.approx(41 / 58)
    for col in ["cv_accuracy_mean", "cv_accuracy_std", "test_accuracy", "test_precision",
                "test_recall", "test_f1"]:
        assert comp[col].dropna().between(0, 1).all()


def test_only_winner_and_baseline_are_evaluated_on_test(trained_artifacts):
    comp = trained_artifacts["result"].comparison
    test_cols = [c for c in comp.columns if c.startswith("test_")]
    assert test_cols
    evaluated = comp["selected"] | comp["is_baseline"]
    assert comp["evaluated_on_test"].equals(evaluated)
    assert comp.loc[evaluated, test_cols].notna().all(axis=None)
    assert comp.loc[~evaluated, test_cols].isna().all(axis=None)
    assert len(comp.loc[~evaluated]) == 1  # um candidato não selecionado nesta execução

    csv = pd.read_csv(trained_artifacts["reports_dir"] / config.COMPARISON_CSV_FILE)
    assert csv.loc[~(csv["selected"] | csv["is_baseline"]), test_cols].isna().all(axis=None)


def test_test_set_is_consulted_only_after_selection(tmp_path, monkeypatch):
    """Registra cada cálculo de métricas: com o conjunto de teste (58 registros) deve
    haver exatamente duas chamadas (vencedor e referência), ambas após a seleção."""
    import src.training as training
    from tests.conftest import fast_specs

    events: list[str] = []
    real_metrics = training.compute_metrics
    real_select = training.select_best_model

    def spy_metrics(y_true, y_pred):
        events.append("test" if len(y_true) == 58 else "train")
        return real_metrics(y_true, y_pred)

    def spy_select(*args, **kwargs):
        events.append("select")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(training, "compute_metrics", spy_metrics)
    monkeypatch.setattr(training, "select_best_model", spy_select)
    training.run_training(
        models_dir=tmp_path / "models", reports_dir=tmp_path / "reports", specs=fast_specs(),
        cv=make_cv(n_repeats=1), n_jobs=1,
    )
    assert events.count("test") == 2
    assert events.index("select") < events.index("test")


def test_selection_ignores_test_metrics(trained_artifacts):
    comp = trained_artifacts["result"].comparison
    expected, _ = select_best_model(comp)
    assert expected == trained_artifacts["result"].best_model_name


def test_training_is_deterministic(trained_artifacts):
    """Repetir a avaliação do modelo selecionado com a mesma semente dá o mesmo resultado."""
    from sklearn.base import clone

    X, y = split_features_target(load_dataset())
    X_tr, X_te, y_tr, _ = split_train_test(X, y)
    model = trained_artifacts["result"].best_pipeline
    refit = clone(model).fit(X_tr, y_tr)
    assert np.array_equal(refit.predict(X_te), model.predict(X_te))


def test_model_and_reports_are_persisted(trained_artifacts):
    reports = trained_artifacts["reports_dir"]
    for name in [config.COMPARISON_CSV_FILE, config.CLASSIFICATION_REPORT_FILE,
                 config.CONFUSION_MATRIX_PLOT_FILE, config.ACCURACY_PLOT_FILE,
                 config.TRAINING_SUMMARY_FILE, config.SMOTE_COMPARISON_FILE]:
        assert (reports / name).is_file(), name
    assert trained_artifacts["model_path"].is_file()

    summary = json.loads((reports / config.TRAINING_SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["random_state"] == config.RANDOM_STATE
    assert summary["selected_model"] == trained_artifacts["result"].best_model_name
    assert summary["n_test"] == 58
    assert sum(map(sum, summary["test_confusion_matrix"]["matrix"])) == 58


def test_saved_pipeline_reloads_and_predicts_identically(trained_artifacts):
    X, y = split_features_target(load_dataset())
    _, X_te, _, _ = split_train_test(X, y)
    loaded = joblib.load(trained_artifacts["model_path"])
    assert isinstance(loaded, ImbPipeline)
    assert [n for n, _ in loaded.steps] == [PREPROCESSOR_STEP, SAMPLER_STEP, "classifier"]
    assert "sampler" in json.loads(
        trained_artifacts["result"].comparison.query("selected")["best_params"].iloc[0])
    assert list(loaded.feature_names_in_) == list(config.FEATURES)
    original = trained_artifacts["result"].best_pipeline
    assert np.array_equal(loaded.predict(X_te), original.predict(X_te))
    # O pipeline salvo aceita dados brutos com valores ausentes (pré-processamento embutido).
    raw_row = X.loc[X["node-caps"].isna()].head(1)
    assert loaded.predict(raw_row)[0] in config.CLASS_LABELS
