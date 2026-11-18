"""Leitura, validacao e EDA da base de credit scoring.

Execucao: .\\.venv\\Scripts\\python.exe src\\train_eval.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DATA_PATH_FALLBACK,
    DATE_COL,
    FEATURES,
    FORBIDDEN_FEATURES,
    GROUP_COL,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
    TARGET,
)

_BOOL_MAP = {"true": 1, "false": 0, "1": 1, "0": 0}

COLOR_MAIN = "#2a78d6"
COLOR_REF = "#52514e"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "azul_cinza_vermelho", ["#2a78d6", "#f0efec", "#e34948"]
)
DISCRETE_NUMERIC = {"qtd_filhos", "qt_pessoas_residencia"}


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


# ---------------------------------------------------------------- EDA
# A EDA e descritiva sobre a base inteira: nada daqui entra no pipeline de modelagem
# e nenhum outlier e removido.


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "savefig.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = REPORTS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def monthly_target_rate(df: pd.DataFrame) -> pd.Series:
    return df.groupby(DATE_COL)[TARGET].mean().sort_index()


def missing_share(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values()


def plot_target_by_month(df: pd.DataFrame) -> Path:
    rate = monthly_target_rate(df)
    overall = df[TARGET].mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(rate.index, rate.values, color=COLOR_MAIN, lw=2, marker="o", ms=5)
    ax.axhline(overall, color=COLOR_REF, ls="--", lw=1)
    ax.annotate(f"media geral {overall:.2%}", (0, overall), xytext=(0, 6),
                textcoords="offset points", color=COLOR_REF, fontsize=9)
    ax.set_title(f"Taxa de {TARGET}=1 por {DATE_COL}")
    ax.set_xlabel(DATE_COL)
    ax.set_ylabel(f"taxa de {TARGET}=1")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.tick_params(axis="x", rotation=60)
    return save_figure(fig, "eda_target_mes.png")


def plot_missing(df: pd.DataFrame) -> Path:
    share = missing_share(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(share.index, share.values, color=COLOR_MAIN, height=0.6)
    for col, value in share.items():
        if value > 0:
            ax.text(value, col, f" {value:.2%}", va="center", fontsize=9)
    ax.set_title("Percentual de ausentes por coluna")
    ax.set_xlabel("ausentes")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, max(share.max() * 1.2, 0.01))
    ax.grid(axis="y", visible=False)
    return save_figure(fig, "eda_faltantes.png")


def plot_numeric(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(4 * len(NUMERIC_FEATURES), 3.8))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        values = df[col].dropna()
        label = col
        if col == "renda":
            # Renda e muito assimetrica; log10 so na visualizacao (exige renda > 0).
            values = np.log10(values[values > 0])
            label = "log10(renda)"
        sns.histplot(values, ax=ax, color=COLOR_MAIN, discrete=col in DISCRETE_NUMERIC,
                     edgecolor=SURFACE)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("linhas" if ax is axes[0] else "")
    fig.suptitle("Distribuicao das numericas (sem remocao de outliers)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    return save_figure(fig, "eda_numericas.png")


def plot_categorical(df: pd.DataFrame) -> Path:
    overall = df[TARGET].mean()
    ncols = 2
    nrows = int(np.ceil(len(CATEGORICAL_FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    flat = axes.ravel()
    for ax, col in zip(flat, CATEGORICAL_FEATURES):
        stats = (
            df[col].fillna("(ausente)").to_frame().assign(y=df[TARGET])
            .groupby(col)["y"].agg(["mean", "size"]).sort_values("mean")
        )
        labels = [f"{cat} (n={size:,})".replace(",", ".")
                  for cat, size in zip(stats.index, stats["size"])]
        ax.barh(labels, stats["mean"], color=COLOR_MAIN, height=0.6)
        ax.axvline(overall, color=COLOR_REF, ls="--", lw=1)
        ax.set_title(col)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    for ax in flat[len(CATEGORICAL_FEATURES):]:
        ax.set_visible(False)
    fig.suptitle(f"Taxa de {TARGET}=1 por categoria (tracejado = media geral {overall:.2%})",
                 fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "eda_categoricas.png")


def plot_correlation(df: pd.DataFrame) -> Path:
    corr = df[NUMERIC_FEATURES + [TARGET]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, ax=ax, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True,
                fmt=".2f", square=True, linewidths=2, linecolor=SURFACE,
                cbar_kws={"label": "correlacao de Pearson"})
    ax.set_title(f"Correlacao entre numericas e {TARGET}")
    return save_figure(fig, "eda_correlacao.png")


def eda_facts(df: pd.DataFrame) -> dict:
    rate = df[TARGET].mean()
    monthly = monthly_target_rate(df)
    missing = missing_share(df)
    return {
        "linhas": int(len(df)),
        "taxa_alvo": round(float(rate), 6),
        "n_mau": int(df[TARGET].sum()),
        # Referencias ingenuas para comparar os modelos depois.
        "baseline_acuracia_sempre_0": round(float(1 - rate), 6),
        "baseline_f1_sempre_1": round(float(2 * rate / (1 + rate)), 6),
        "ausentes": {c: round(float(v), 6) for c, v in missing.items() if v > 0},
        "drift_taxa_mensal_min": {"mes": str(monthly.idxmin()),
                                  "taxa": round(float(monthly.min()), 6)},
        "drift_taxa_mensal_max": {"mes": str(monthly.idxmax()),
                                  "taxa": round(float(monthly.max()), 6)},
        "n_meses": int(monthly.size),
        "n_clientes": int(df[GROUP_COL].nunique()),
        "duplicatas_exatas": int(df.duplicated().sum()),
        "duplicatas_sem_unnamed0": int(df.drop(columns="Unnamed: 0").duplicated().sum()),
    }


def run_eda(df: pd.DataFrame) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    paths = [
        plot_target_by_month(df),
        plot_missing(df),
        plot_numeric(df),
        plot_categorical(df),
        plot_correlation(df),
    ]
    for path in paths:
        print(f"Salvo {path.relative_to(REPORTS_DIR.parent)}")
    facts = eda_facts(df)
    facts_path = REPORTS_DIR / "eda_fatos.json"
    facts_path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo {facts_path.relative_to(REPORTS_DIR.parent)}")
    return facts


def main() -> None:
    df = load_data()
    X, y = split_features_target(df)
    report(df, X, y)
    facts = run_eda(df)
    print("Fatos da EDA:")
    for key, value in facts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
