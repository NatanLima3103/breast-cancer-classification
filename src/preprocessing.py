"""Pipeline de pré-processamento (Scikit-learn + imbalanced-learn).

O pré-processamento é sempre embutido no ``Pipeline`` de cada modelo. Assim,
imputadores, codificadores e escalonadores são ajustados apenas com os dados de
treino de cada divisão (de cada *fold* da validação cruzada e, no ajuste final,
apenas com o conjunto de treinamento), o que evita vazamento de dados.

O pipeline do modelo é um ``imblearn.pipeline.Pipeline`` com uma etapa opcional de
balanceamento (``sampler``). O sampler (por exemplo, SMOTE) é aplicado somente no
``fit``, ou seja, só nos dados de treino de cada divisão; no ``predict`` a etapa é
ignorada, de modo que o teste e as novas pacientes nunca são reamostrados.
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src import config

PREPROCESSOR_STEP = "preprocessor"
SAMPLER_STEP = "sampler"
CLASSIFIER_STEP = "classifier"
NO_SAMPLER = "passthrough"
CATEGORICAL_BRANCH = "categorical"
ORDINAL_BRANCH = "ordinal"


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Constrói o ``ColumnTransformer`` dos nove atributos preditores.

    * Categóricos nominais: imputação pela moda + ``OneHotEncoder``
      (``handle_unknown="ignore"``: categorias não vistas viram um vetor de zeros;
      a camada de inferência só permite isso explicitamente, ver ``inference.py``).
    * ``deg-malig``: mantido como número (1 < 2 < 3), preservando a ordem natural.
      Recebe imputação pela moda e, para modelos sensíveis à escala, ``MinMaxScaler``
      que o leva ao intervalo [0, 1], o mesmo das variáveis binárias do one-hot.
    """
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    ordinal_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
    if scale:
        ordinal_steps.append(("scaler", MinMaxScaler()))
    ordinal = Pipeline(steps=ordinal_steps)

    return ColumnTransformer(
        transformers=[
            (CATEGORICAL_BRANCH, categorical, list(config.CATEGORICAL_FEATURES)),
            (ORDINAL_BRANCH, ordinal, list(config.ORDINAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    classifier: BaseEstimator, scale: bool, sampler: BaseEstimator | str = NO_SAMPLER
) -> ImbPipeline:
    """Combina pré-processamento, balanceamento opcional e classificador.

    ``sampler`` é ``"passthrough"`` (sem balanceamento) por padrão; a busca de
    hiperparâmetros pode trocá-lo por ``SMOTE``. O imblearn aplica o sampler só no
    ``fit`` (treino de cada divisão) e o ignora no ``predict``.
    """
    return ImbPipeline(
        steps=[
            (PREPROCESSOR_STEP, build_preprocessor(scale=scale)),
            (SAMPLER_STEP, sampler),
            (CLASSIFIER_STEP, classifier),
        ]
    )


def get_training_categories(pipeline: Pipeline) -> dict[str, list]:
    """Retorna as categorias nominais aprendidas pelo ``OneHotEncoder`` no ajuste.

    Usado pela inferência para detectar categorias não vistas no treinamento.
    ``deg-malig`` não aparece aqui: é numérico e validado pelo domínio documentado.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps[PREPROCESSOR_STEP]
    categorical: Pipeline = preprocessor.named_transformers_[CATEGORICAL_BRANCH]
    encoder: OneHotEncoder = categorical.named_steps["encoder"]
    categories = {
        feature: [str(v) for v in cats]
        for feature, cats in zip(config.CATEGORICAL_FEATURES, encoder.categories_)
    }
    return categories
