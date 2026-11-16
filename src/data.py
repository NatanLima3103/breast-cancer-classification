"""Leitura, validação de esquema e separação da base de dados."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


class DataValidationError(ValueError):
    """Erro levantado quando a base não segue o esquema esperado."""


def read_raw_dataset(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Lê o CSV exatamente como está no arquivo (todas as colunas como texto).

    O marcador ``?`` é preservado, o que permite à análise exploratória contar
    os valores desconhecidos originais.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo da base de dados não encontrado: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.astype(object)
    missing_cols = [c for c in config.ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Colunas ausentes na base: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in config.ALL_COLUMNS]
    if extra_cols:
        raise DataValidationError(f"Colunas inesperadas na base: {extra_cols}")
    return df[list(config.ALL_COLUMNS)]


def replace_missing_marker(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o marcador ``?`` em valores ausentes reais (``np.nan``)."""
    return df.replace(config.MISSING_MARKER, np.nan)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica os tipos definitivos: ``deg-malig`` numérico e o restante como objeto."""
    out = df.copy()
    for col in config.ORDINAL_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
    for col in config.CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(object)
    if config.TARGET in out.columns:
        out[config.TARGET] = out[config.TARGET].astype(object)
    return out


def validate_domain(df: pd.DataFrame) -> None:
    """Garante que todo valor não ausente pertença ao domínio documentado."""
    problems: list[str] = []
    for col in df.columns:
        allowed = set(config.DOCUMENTED_DOMAIN[col])
        if col in config.ORDINAL_FEATURES:
            allowed = {float(v) for v in allowed}
        observed = set(df[col].dropna().unique())
        invalid = observed - allowed
        if invalid:
            problems.append(f"{col}: {sorted(map(str, invalid))}")
    if config.TARGET in df.columns and df[config.TARGET].isna().any():
        problems.append(f"{config.TARGET}: a variável-alvo possui valores ausentes")
    if problems:
        raise DataValidationError("Valores fora do domínio documentado: " + "; ".join(problems))


def load_dataset(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Lê a base, converte ``?`` em ausentes, ajusta tipos e valida o domínio."""
    df = coerce_types(replace_missing_marker(read_raw_dataset(path)))
    validate_domain(df)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa os nove atributos preditores (``X``) da variável-alvo ``Class`` (``y``)."""
    if config.TARGET not in df.columns:
        raise DataValidationError(f"A coluna-alvo '{config.TARGET}' não está presente.")
    X = df[list(config.FEATURES)].copy()
    y = df[config.TARGET].copy()
    return X, y
