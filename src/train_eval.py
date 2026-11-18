"""Leitura, validacao e EDA da base de credit scoring.

Execucao: .\\.venv\\Scripts\\python.exe src\\train_eval.py
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DATA_PATH_FALLBACK,
    DATE_COL,
    FEATURES,
    FORBIDDEN_FEATURES,
    GROUP_COL,
    N_TEST,
    N_TRAIN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
    TARGET,
    TEST_START,
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


# ------------------------------------------------- split temporal e pre-processamento


def sample_stratified(df: pd.DataFrame, n: int) -> pd.DataFrame:
    strata = df[DATE_COL].astype(str) + "_" + df[TARGET].astype(str)
    sample, _ = train_test_split(df, train_size=n, stratify=strata, random_state=RANDOM_STATE)
    return sample.sort_index()


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Filtra o periodo antes de amostrar para que nenhuma linha de 2016 entre no treino.
    dates = pd.to_datetime(df[DATE_COL])
    cutoff = pd.Timestamp(TEST_START)
    train = sample_stratified(df[dates < cutoff], N_TRAIN)
    test = sample_stratified(df[dates >= cutoff], N_TEST)
    assert pd.to_datetime(train[DATE_COL]).max() < cutoff <= pd.to_datetime(test[DATE_COL]).min()
    return train, test


def report_split(train: pd.DataFrame, test: pd.DataFrame, df: pd.DataFrame) -> None:
    for name, part in (("Treino", train), ("Teste", test)):
        print(f"{name}: {len(part):,} linhas | {part[DATE_COL].min()} a {part[DATE_COL].max()} | "
              f"mau=1 {int(part[TARGET].sum()):,} ({part[TARGET].mean():.2%}) | "
              f"clientes {part[GROUP_COL].nunique():,}")
    shared = set(train[GROUP_COL]) & set(test[GROUP_COL])
    print(f"Clientes nas duas amostras (treino e teste): {len(shared):,} "
          f"({len(shared) / test[GROUP_COL].nunique():.2%} dos clientes do teste)")
    dates = pd.to_datetime(df[DATE_COL])
    cutoff = pd.Timestamp(TEST_START)
    shared_full = set(df.loc[dates < cutoff, GROUP_COL]) & set(df.loc[dates >= cutoff, GROUP_COL])
    print(f"Clientes nos dois periodos (base completa, 2015 e 2016): {len(shared_full):,}")


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # remainder="drop": index, data_ref, Unnamed: 0 e o alvo nunca chegam ao modelo.
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def check_preprocessor(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    # Ajuste so no treino; o teste passa apenas por transform para conferir o formato.
    preprocessor = build_preprocessor().fit(X_train)
    names = list(preprocessor.get_feature_names_out())
    leaked = [n for n in names if any(col in n for col in FORBIDDEN_FEATURES)]
    assert not leaked, f"Colunas proibidas no pre-processamento: {leaked}"
    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)
    assert not np.isnan(Xt_train).any() and not np.isnan(Xt_test).any()
    medians = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_
    print(f"Pre-processamento ajustado no treino: {len(names)} colunas de saida")
    print(f"  shape treino {Xt_train.shape} | shape teste (so transform) {Xt_test.shape}")
    print(f"  medianas do treino: {dict(zip(NUMERIC_FEATURES, np.round(medians, 4)))}")
    print(f"  colunas: {names}")


# ------------------------------------------------------------------ experimentos


@dataclass
class Experiment:
    name: str
    class_weight: str
    pipeline: Pipeline
    fit_seconds: float = 0.0


def build_experiments() -> list[Experiment]:
    experiments = []
    for weight in (None, "balanced"):
        label = "balanced" if weight else "None"
        models = {
            "regressao_logistica": LogisticRegression(
                max_iter=2000, class_weight=weight, random_state=RANDOM_STATE),
            "arvore_decisao": DecisionTreeClassifier(
                max_depth=4, class_weight=weight, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=300, n_jobs=-1, class_weight=weight, random_state=RANDOM_STATE),
        }
        experiments += [Experiment(name, label, build_pipeline(model))
                        for name, model in models.items()]
    # KNN nao tem class_weight nem random_state.
    knn = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    experiments.append(Experiment("knn", "nao_aplicavel", build_pipeline(knn)))
    return experiments


def fit_experiments(X_train: pd.DataFrame, y_train: pd.Series) -> list[Experiment]:
    # Modelos ficam so em memoria: nenhum e salvo em disco (Random Forests ocupam muito espaco).
    experiments = build_experiments()
    for exp in experiments:
        start = time.perf_counter()
        exp.pipeline.fit(X_train, y_train)
        exp.fit_seconds = time.perf_counter() - start
        print(f"  {exp.name:<20} class_weight={exp.class_weight:<14} "
              f"ajuste {exp.fit_seconds:6.1f}s")
    return experiments


def check_experiments(experiments: list[Experiment], X_train: pd.DataFrame) -> None:
    assert len(experiments) == 7, f"Esperados 7 experimentos, obtidos {len(experiments)}"
    for exp in experiments:
        proba = exp.pipeline.predict_proba(X_train)[:, 1]
        assert proba.shape == (len(X_train),) and ((proba >= 0) & (proba <= 1)).all()
    print(f"{len(experiments)} experimentos ajustados no treino e mantidos em memoria; "
          f"predict_proba com shape ({len(X_train):,},) e valores em [0, 1]")


def main() -> None:
    df = load_data()
    X, y = split_features_target(df)
    report(df, X, y)
    facts = run_eda(df)
    print("Fatos da EDA:")
    for key, value in facts.items():
        print(f"  {key}: {value}")

    train, test = temporal_split(df)
    report_split(train, test, df)
    X_train, y_train = split_features_target(train)
    X_test, _ = split_features_target(test)
    check_preprocessor(X_train, X_test)

    print("Experimentos (ajuste no treino 2015):")
    experiments = fit_experiments(X_train, y_train)
    check_experiments(experiments, X_train)


if __name__ == "__main__":
    main()
