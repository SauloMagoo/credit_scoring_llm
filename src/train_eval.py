"""Leitura e validacao da base de credit scoring.

Execucao: .\\.venv\\Scripts\\python.exe src\\train_eval.py
"""
from pathlib import Path

import pandas as pd

from config import (
    DATA_PATH,
    DATA_PATH_FALLBACK,
    FEATURES,
    FORBIDDEN_FEATURES,
    GROUP_COL,
    REQUIRED_COLUMNS,
    TARGET,
)

_BOOL_MAP = {"true": 1, "false": 0, "1": 1, "0": 0}


def resolve_data_path() -> Path:
    for path in (DATA_PATH, DATA_PATH_FALLBACK):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Base nao encontrada. Esperado em {DATA_PATH} ou {DATA_PATH_FALLBACK}."
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")


def target_to_int(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Alvo '{TARGET}' tem {series.isna().sum()} valores ausentes.")
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    mapped = series.astype(str).str.strip().str.lower().map(_BOOL_MAP)
    if mapped.isna().any():
        invalid = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Valores invalidos no alvo '{TARGET}': {invalid}")
    return mapped.astype("int8")


def load_data() -> pd.DataFrame:
    path = resolve_data_path()
    print(f"Lendo {path}")
    df = pd.read_csv(path, encoding="utf-8")
    validate_columns(df)
    df[TARGET] = target_to_int(df[TARGET])
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    leaked = set(FEATURES) & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"Colunas proibidas na lista de features: {sorted(leaked)}")
    X = df[FEATURES].copy()
    assert not set(X.columns) & set(FORBIDDEN_FEATURES)
    return X, df[TARGET]


def report(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    rows_without_rownum = df.drop(columns="Unnamed: 0")
    print(f"Shape da base: {df.shape}")
    print(f"Clientes distintos ({GROUP_COL}): {df[GROUP_COL].nunique():,}")
    print(f"Duplicatas exatas (todas as colunas): {df.duplicated().sum():,}")
    print(f"Duplicatas ignorando 'Unnamed: 0': {rows_without_rownum.duplicated().sum():,}")
    print(f"mau=1: {int(y.sum()):,} de {len(y):,} ({y.mean():.2%})")
    print(f"Ausentes em tempo_emprego: {df['tempo_emprego'].isna().mean():.2%}")
    print(f"Features ({X.shape[1]}): {list(X.columns)}")
    print(f"Fora das features: {sorted(set(df.columns) - set(X.columns))}")


def main() -> None:
    df = load_data()
    X, y = split_features_target(df)
    report(df, X, y)


if __name__ == "__main__":
    main()
