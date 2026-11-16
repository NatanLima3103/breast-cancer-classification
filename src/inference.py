"""Módulo de inferência: classifica uma nova paciente com o modelo treinado.

Uso:
    python -m src.inference --json examples/nova_paciente.json
    python -m src.inference --age 50-59 --menopause ge40 --tumor-size 15-19 \\
        --inv-nodes 0-2 --node-caps no --deg-malig 1 --breast right \\
        --breast-quad central --irradiat no

AVISO: o resultado é uma classificação experimental baseada em uma base histórica
e não constitui diagnóstico médico nem substitui avaliação clínica.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import get_training_categories

DISCLAIMER = (
    "Aviso: classificação experimental baseada em uma base histórica (UCI, 1988). "
    "Não é diagnóstico médico e não substitui avaliação clínica."
)
# Valores interpretados como "desconhecido" (equivalentes ao '?' da base original).
UNKNOWN_VALUES = (None, config.MISSING_MARKER)


class InputValidationError(ValueError):
    """Entrada da paciente inválida (atributos ausentes, extras ou valores inválidos)."""


class ModelLoadError(RuntimeError):
    """Modelo ausente, corrompido ou incompatível com o esquema de entrada."""


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    probabilities: dict[str, float] | None


# ---------------------------------------------------------------------------
# Leitura e validação da entrada
# ---------------------------------------------------------------------------
def load_patient_json(path: Path | str) -> dict[str, Any]:
    """Lê o arquivo JSON com os atributos de uma paciente."""
    path = Path(path)
    if not path.is_file():
        raise InputValidationError(f"Arquivo JSON não encontrado: {path}")
    try:
        # "utf-8-sig" aceita UTF-8 com e sem BOM (o PowerShell 5.1 grava com BOM).
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"JSON inválido em {path}: {exc.msg} (linha {exc.lineno}, coluna {exc.colno})"
        ) from exc
    if not isinstance(data, dict):
        raise InputValidationError(
            "O JSON deve conter um único objeto com os nove atributos da paciente "
            f"(recebido: {type(data).__name__})."
        )
    return data


def _validate_deg_malig(value: Any) -> float:
    allowed = config.DOCUMENTED_DOMAIN["deg-malig"]
    number: float | None = None
    if isinstance(value, bool):
        number = None
    elif isinstance(value, int):
        number = float(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = float(value.strip())
    if number is None or int(number) not in allowed:
        raise InputValidationError(
            f"'deg-malig' inválido: {value!r}. Valores permitidos: {list(allowed)}."
        )
    return number


def validate_patient(
    record: dict[str, Any],
    training_categories: dict[str, list[str]] | None = None,
    allow_unseen_categories: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Valida os atributos de uma paciente e devolve um DataFrame de uma linha.

    Regras:
    * Os nove atributos preditores são obrigatórios; chaves extras (inclusive
      ``Class``) são rejeitadas.
    * Valores devem pertencer ao domínio documentado pela UCI.
    * ``null`` ou ``"?"`` indicam valor desconhecido: são imputados pelo pipeline
      (moda do treino) e geram um aviso.
    * Categorias válidas no domínio, mas não vistas no treinamento, são rejeitadas
      por padrão, pois o ``OneHotEncoder`` as representaria como um vetor de zeros
      (o atributo seria efetivamente ignorado). Só são aceitas com
      ``allow_unseen_categories=True`` e geram aviso.

    Todos os problemas encontrados são reunidos em uma única exceção.
    """
    if not isinstance(record, dict):
        raise InputValidationError("A entrada da paciente deve ser um objeto JSON (dicionário).")

    errors: list[str] = []
    warnings: list[str] = []
    missing = [f for f in config.FEATURES if f not in record]
    if missing:
        errors.append(f"Atributos ausentes: {missing}.")
    extra = [k for k in record if k not in config.FEATURES]
    if config.TARGET in extra:
        errors.append(
            f"A variável-alvo '{config.TARGET}' não deve ser enviada para inferência."
        )
        extra.remove(config.TARGET)
    if extra:
        errors.append(f"Atributos não reconhecidos: {extra}. Esperados: {list(config.FEATURES)}.")

    values: dict[str, Any] = {}
    for feature in config.FEATURES:
        if feature not in record:
            continue
        raw = record[feature]
        if isinstance(raw, str):
            raw = raw.strip()
        if raw in UNKNOWN_VALUES:
            values[feature] = np.nan
            warnings.append(
                f"'{feature}' informado como desconhecido; será imputado com a categoria "
                "mais frequente do treinamento."
            )
            continue
        try:
            if feature in config.ORDINAL_FEATURES:
                values[feature] = _validate_deg_malig(raw)
                continue
            allowed = config.DOCUMENTED_DOMAIN[feature]
            if not isinstance(raw, str) or raw not in allowed:
                raise InputValidationError(
                    f"'{feature}' inválido: {raw!r}. Valores permitidos: {list(allowed)}."
                )
            seen = training_categories.get(feature) if training_categories else None
            if seen is not None and raw not in seen:
                message = (
                    f"'{feature}'={raw!r} pertence ao domínio documentado, mas não ocorreu no "
                    f"treinamento (vistas: {seen}); o modelo ignoraria este atributo."
                )
                if not allow_unseen_categories:
                    raise InputValidationError(
                        message + " Use --allow-unseen-categories para aceitar mesmo assim."
                    )
                warnings.append(message)
            values[feature] = raw
        except InputValidationError as exc:
            errors.append(str(exc))

    if errors:
        raise InputValidationError("Entrada inválida:\n  - " + "\n  - ".join(errors))

    frame = pd.DataFrame([values], columns=list(config.FEATURES))
    for col in config.CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype(object)
    frame[list(config.ORDINAL_FEATURES)] = frame[list(config.ORDINAL_FEATURES)].astype(float)
    return frame, warnings


# ---------------------------------------------------------------------------
# Modelo e previsão
# ---------------------------------------------------------------------------
def load_model(path: Path | str = config.MODEL_PATH) -> Pipeline:
    """Carrega o pipeline salvo e verifica sua compatibilidade com o esquema."""
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"Arquivo do modelo não encontrado: {path}. "
            "Execute o treinamento antes: python -m src.training"
        )
    try:
        model = joblib.load(path)
    except Exception as exc:  # erro de desserialização: reportado, nunca ignorado
        raise ModelLoadError(f"Não foi possível carregar o modelo em {path}: {exc}") from exc
    if not isinstance(model, Pipeline) or not hasattr(model, "predict"):
        raise ModelLoadError(f"O arquivo {path} não contém um Pipeline do scikit-learn.")
    expected = list(config.FEATURES)
    fitted_features = list(getattr(model, "feature_names_in_", []))
    if fitted_features != expected:
        raise ModelLoadError(
            "O modelo salvo foi treinado com atributos diferentes dos esperados: "
            f"{fitted_features} (esperado: {expected})."
        )
    return model


def predict(model: Pipeline, patient: pd.DataFrame) -> PredictionResult:
    """Executa pré-processamento + classificação. As probabilidades só são
    retornadas quando o estimador oferece ``predict_proba``.
    """
    label = str(model.predict(patient)[0])
    probabilities: dict[str, float] | None = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(patient)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return PredictionResult(predicted_class=label, probabilities=probabilities)


def classify_patient(
    record: dict[str, Any],
    model_path: Path | str = config.MODEL_PATH,
    allow_unseen_categories: bool = False,
) -> tuple[PredictionResult, list[str]]:
    """Fluxo completo: carrega o modelo, valida a entrada e faz a previsão."""
    model = load_model(model_path)
    patient, warnings = validate_patient(
        record, get_training_categories(model), allow_unseen_categories
    )
    return predict(model, patient), warnings


def format_result(result: PredictionResult, warnings: list[str] | None = None) -> str:
    lines = ["Resultado da inferência", f"Classe prevista: {result.predicted_class}"]
    if result.probabilities is not None:
        lines.append("Probabilidades estimadas:")
        for label in config.CLASS_LABELS:
            if label in result.probabilities:
                lines.append(f"  * {label}: {result.probabilities[label]:.2%}")
    else:
        lines.append("Probabilidades estimadas: não disponíveis para o modelo selecionado "
                     "(o estimador não implementa predict_proba).")
    for warning in warnings or []:
        lines.append(f"Atenção: {warning}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.inference",
        description="Classifica uma nova paciente com o modelo salvo em models/best_model.joblib. "
                    "Informe os atributos por --json ou pelos nove parâmetros individuais.",
        epilog=DISCLAIMER,
    )
    parser.add_argument("--json", type=Path, help="Arquivo JSON com os nove atributos.")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH,
                        help="Caminho do modelo (padrão: models/best_model.joblib).")
    parser.add_argument("--allow-unseen-categories", action="store_true",
                        help="Aceita categorias válidas no domínio da UCI, mas não vistas no "
                             "treinamento (o atributo passa a ser ignorado pelo modelo).")
    group = parser.add_argument_group("atributos individuais (alternativa ao --json)")
    for feature in config.FEATURES:
        group.add_argument(f"--{feature}", dest=feature.replace("-", "_"), metavar="VALOR",
                           help=config.FEATURE_DESCRIPTIONS[feature])
    return parser


def _record_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    individual = {f: getattr(args, f.replace("-", "_")) for f in config.FEATURES}
    provided = {k: v for k, v in individual.items() if v is not None}
    if args.json is not None:
        if provided:
            parser.error("use --json OU os atributos individuais, não ambos.")
        return load_patient_json(args.json)
    if not provided:
        parser.error(
            "nenhuma paciente informada. Exemplo: "
            "python -m src.inference --json examples/nova_paciente.json"
        )
    # Atributos individuais ausentes são mantidos fora do dicionário para que a
    # validação os reporte explicitamente.
    return provided


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        record = _record_from_args(args, parser)
        result, warnings = classify_patient(record, args.model, args.allow_unseen_categories)
    except (InputValidationError, ModelLoadError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    print(format_result(result, warnings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
