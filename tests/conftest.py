"""Fixtures compartilhadas pelos testes.

Os testes usam apenas a base fornecida em ``data/`` e diretórios temporários
criados pelo próprio Pytest (``tmp_path``); nenhum artefato preexistente é exigido.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

from src import config  # noqa: E402
from src.training import (  # noqa: E402
    BASELINE_NAME,
    ModelSpec,
    _prefixed,
    get_model_specs,
    run_training,
)


@pytest.fixture
def example_patient() -> dict:
    return {
        "age": "50-59",
        "menopause": "ge40",
        "tumor-size": "15-19",
        "inv-nodes": "0-2",
        "node-caps": "no",
        "deg-malig": 1,
        "breast": "right",
        "breast-quad": "central",
        "irradiat": "no",
    }


def fast_specs() -> list[ModelSpec]:
    """Subconjunto pequeno e determinístico dos modelos, para testes rápidos."""
    by_name = {spec.name: spec for spec in get_model_specs()}
    return [
        by_name[BASELINE_NAME],
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE),
            param_grid=_prefixed({"C": [0.1, 1.0]}),  # inclui sampler: passthrough/SMOTE
            scale=True,
        ),
        by_name["Gaussian Naive Bayes"],
    ]


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Executa um treinamento reduzido (uma vez por sessão) em diretórios temporários."""
    base = tmp_path_factory.mktemp("training")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=config.RANDOM_STATE)
    result = run_training(
        data_path=config.DATA_PATH,
        models_dir=base / "models",
        reports_dir=base / "reports",
        specs=fast_specs(),
        cv=cv,
        n_jobs=1,
    )
    return {
        "result": result,
        "models_dir": base / "models",
        "reports_dir": base / "reports",
        "model_path": base / "models" / config.MODEL_PATH.name,
    }
