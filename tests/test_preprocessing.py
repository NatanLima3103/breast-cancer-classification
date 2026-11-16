"""Testes de leitura da base e do pré-processamento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data import (
    DataValidationError,
    load_dataset,
    read_raw_dataset,
    replace_missing_marker,
    split_features_target,
)
from src.preprocessing import (
    CATEGORICAL_BRANCH,
    NO_SAMPLER,
    ORDINAL_BRANCH,
    PREPROCESSOR_STEP,
    SAMPLER_STEP,
    build_pipeline,
    build_preprocessor,
    get_training_categories,
)


# --- Leitura da base ---------------------------------------------------------
def test_raw_dataset_shape_and_columns():
    raw = read_raw_dataset()
    assert raw.shape == (286, 10)
    assert list(raw.columns) == list(config.ALL_COLUMNS)


def test_raw_dataset_known_characteristics():
    raw = read_raw_dataset()
    counts = raw[config.TARGET].value_counts()
    assert counts[config.NEGATIVE_CLASS] == 201
    assert counts[config.POSITIVE_CLASS] == 85
    unknown = (raw == config.MISSING_MARKER).sum()
    assert unknown["node-caps"] == 8
    assert unknown["breast-quad"] == 1
    assert unknown.drop(["node-caps", "breast-quad"]).sum() == 0
    assert raw.duplicated().sum() == 14
    assert raw[list(config.FEATURES)].duplicated().sum() == 20


def test_missing_marker_converted_to_nan():
    df = load_dataset()
    assert not (df == config.MISSING_MARKER).any().any()
    assert df["node-caps"].isna().sum() == 8
    assert df["breast-quad"].isna().sum() == 1
    assert len(df) == 286  # nenhum registro removido


def test_replace_missing_marker_only_touches_question_marks():
    frame = pd.DataFrame({"a": ["?", "x", "?y"]})
    out = replace_missing_marker(frame)
    assert pd.isna(out.loc[0, "a"])
    assert out.loc[1, "a"] == "x"
    assert out.loc[2, "a"] == "?y"


def test_deg_malig_is_numeric_ordinal():
    df = load_dataset()
    assert pd.api.types.is_float_dtype(df["deg-malig"])
    assert sorted(df["deg-malig"].unique()) == [1.0, 2.0, 3.0]


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        read_raw_dataset(tmp_path / "inexistente.csv")


def test_invalid_category_in_dataset_is_rejected(tmp_path):
    raw = read_raw_dataset()
    raw.loc[0, "menopause"] = "invalido"
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="menopause"):
        load_dataset(path)


# --- Separação X / y ---------------------------------------------------------
def test_split_features_target():
    X, y = split_features_target(load_dataset())
    assert list(X.columns) == list(config.FEATURES)
    assert config.TARGET not in X.columns
    assert y.name == config.TARGET
    assert len(X) == len(y) == 286
    assert set(y.unique()) == set(config.CLASS_LABELS)


# --- Pré-processamento --------------------------------------------------------
def test_preprocessor_output_has_no_missing_and_expected_width():
    X, _ = split_features_target(load_dataset())
    pre = build_preprocessor(scale=True)
    Xt = pre.fit_transform(X)
    n_onehot = sum(X[c].nunique(dropna=True) for c in config.CATEGORICAL_FEATURES)
    assert Xt.shape == (286, n_onehot + len(config.ORDINAL_FEATURES))
    assert not np.isnan(Xt).any()


def test_preprocessor_preserves_deg_malig_order_and_scales():
    X, _ = split_features_target(load_dataset())
    pre = build_preprocessor(scale=True).fit(X)
    probe = X.iloc[:3].copy()
    probe["deg-malig"] = [1.0, 2.0, 3.0]
    col = list(pre.get_feature_names_out()).index("deg-malig")
    values = pre.transform(probe)[:, col]
    assert values.tolist() == [0.0, 0.5, 1.0]

    unscaled = build_preprocessor(scale=False).fit(X).transform(probe)[:, col]
    assert unscaled.tolist() == [1.0, 2.0, 3.0]


def test_imputation_uses_most_frequent_category():
    X, _ = split_features_target(load_dataset())
    pre = build_preprocessor(scale=False).fit(X)
    imputer = pre.named_transformers_[CATEGORICAL_BRANCH].named_steps["imputer"]
    idx = list(config.CATEGORICAL_FEATURES).index("node-caps")
    assert imputer.statistics_[idx] == X["node-caps"].mode().iloc[0] == "no"


def test_unknown_category_is_encoded_as_zeros():
    X, _ = split_features_target(load_dataset())
    pre = build_preprocessor(scale=False).fit(X)
    probe = X.iloc[[0]].copy()
    probe["age"] = "90-99"  # não presente no ajuste
    names = list(pre.get_feature_names_out())
    age_cols = [i for i, n in enumerate(names) if n.startswith("age_")]
    assert pre.transform(probe)[0, age_cols].sum() == 0


def test_no_leakage_preprocessor_learns_only_from_training_rows():
    """Imputador e codificador devem refletir somente os dados usados no ajuste."""
    X, _ = split_features_target(load_dataset())
    # "Treino" sem pacientes 30-39, com node-caps="yes" como moda e sem deg-malig=1.
    pool = X.loc[(X["age"] != "30-39") & (X["deg-malig"] != 1.0)]
    train = pd.concat([
        pool.loc[pool["node-caps"] == "yes"],
        pool.loc[pool["node-caps"] == "no"].head(5),
        pool.loc[pool["node-caps"].isna()],
    ])
    assert train["node-caps"].mode().iloc[0] == "yes"
    assert X["node-caps"].mode().iloc[0] == "no"  # moda da base completa é diferente

    pipe = build_pipeline(DummyClassifier(), scale=True)
    pipe.fit(train, ["no-recurrence-events"] * len(train))
    pre = pipe.named_steps[PREPROCESSOR_STEP]
    cat = pre.named_transformers_[CATEGORICAL_BRANCH]
    learned = get_training_categories(pipe)
    assert "30-39" not in learned["age"]
    idx = list(config.CATEGORICAL_FEATURES).index("node-caps")
    assert cat.named_steps["imputer"].statistics_[idx] == "yes"

    scaler = pre.named_transformers_[ORDINAL_BRANCH].named_steps["scaler"]
    assert scaler.data_min_[0] == train["deg-malig"].min() == 2.0  # não 1.0 da base completa
    assert scaler.data_max_[0] == train["deg-malig"].max()


def test_build_pipeline_contains_preprocessing_sampler_and_classifier():
    pipe = build_pipeline(DummyClassifier(), scale=False)
    assert isinstance(pipe, ImbPipeline)
    assert [name for name, _ in pipe.steps] == [PREPROCESSOR_STEP, SAMPLER_STEP, "classifier"]
    assert pipe.named_steps[SAMPLER_STEP] == NO_SAMPLER  # sem balanceamento por padrão


def _synthetic_imbalanced(n_neg: int = 85, n_pos: int = 15, seed: int = 0):
    """Pacientes sintéticas (valores sorteados do domínio observado) com classes desbalanceadas."""
    rng = np.random.default_rng(seed)
    domain = {f: [v for v in config.DOCUMENTED_DOMAIN[f]] for f in config.FEATURES}
    domain["age"] = ["30-39", "40-49", "50-59", "60-69"]
    rows = {f: rng.choice(domain[f], size=n_neg + n_pos) for f in config.FEATURES}
    X = pd.DataFrame(rows, columns=list(config.FEATURES))
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    X["deg-malig"] = X["deg-malig"].astype(float)
    y = pd.Series([config.NEGATIVE_CLASS] * n_neg + [config.POSITIVE_CLASS] * n_pos)
    return X, y


def test_smote_resamples_only_during_fit():
    """Com SMOTE, o classificador é treinado com as classes balanceadas, mas o
    predict não altera a quantidade de linhas (o sampler é ignorado na previsão)."""
    X, y = _synthetic_imbalanced(n_neg=85, n_pos=15)
    pipe = build_pipeline(
        DecisionTreeClassifier(random_state=config.RANDOM_STATE), scale=False,
        sampler=SMOTE(random_state=config.RANDOM_STATE),
    ).fit(X, y)
    tree = pipe.named_steps["classifier"]
    assert tree.tree_.n_node_samples[0] == 2 * 85  # minoria sobreamostrada até 85
    assert len(pipe.predict(X)) == len(X) == 100
    assert len(pipe.predict(X.head(7))) == 7

    plain = build_pipeline(DecisionTreeClassifier(random_state=config.RANDOM_STATE), scale=False)
    plain.fit(X, y)
    assert plain.named_steps["classifier"].tree_.n_node_samples[0] == 100  # sem reamostragem
