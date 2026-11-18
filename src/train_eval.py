"""Leitura, EDA, split temporal, CV agrupada, escolha do modelo e avaliacao no teste.

Execucao: .\\.venv\\Scripts\\python.exe src\\train_eval.py
"""
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib
import sklearn

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.base import BaseEstimator, clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # noqa: E402
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
    MODEL_PATH,
    MODELS_DIR,
    N_APP_SAMPLE,
    N_FOLDS,
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
    oof_proba: np.ndarray | None = None
    threshold: float = 0.5
    cv_seconds: float = 0.0
    oof: dict = field(default_factory=dict)
    oof_05: dict = field(default_factory=dict)
    folds: pd.DataFrame | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.class_weight})"


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


# ------------------------------------------------- CV agrupada, limiar e escolha
# Tudo nesta secao usa so o treino 2015. O teste 2016 so e lido depois da escolha.


def make_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> list[tuple]:
    # Os mesmos folds para todos os experimentos: a comparacao fica pareada.
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y, groups))
    for train_idx, valid_idx in folds:
        shared = set(groups.iloc[train_idx]) & set(groups.iloc[valid_idx])
        assert not shared, f"{len(shared)} clientes em treino e validacao do mesmo fold"
    covered = np.sort(np.concatenate([valid for _, valid in folds]))
    assert np.array_equal(covered, np.arange(len(y))), "Folds nao cobrem o treino uma vez"
    return folds


def classification_metrics(y: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    # Mesma regra de precision_recall_curve: positivo quando proba >= limiar.
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "limiar": float(threshold),
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y: pd.Series, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    # O ultimo ponto da curva (precisao=1, recall=0) nao tem limiar associado.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def run_oof(exp: Experiment, X: pd.DataFrame, y: pd.Series, folds: list[tuple]) -> None:
    proba = np.full(len(y), np.nan)
    fold_id = np.empty(len(y), dtype=int)
    start = time.perf_counter()
    for k, (train_idx, valid_idx) in enumerate(folds):
        model = clone(exp.pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_id[valid_idx] = k
    exp.cv_seconds = time.perf_counter() - start
    assert not np.isnan(proba).any()

    exp.oof_proba = proba
    exp.threshold = best_f1_threshold(y, proba)
    exp.oof = classification_metrics(y, proba, exp.threshold)
    exp.oof_05 = classification_metrics(y, proba, 0.5)
    exp.folds = pd.DataFrame([
        classification_metrics(y[fold_id == k], proba[fold_id == k], exp.threshold)
        for k in range(len(folds))
    ])


def cv_table(experiments: list[Experiment]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        row = {"experimento": exp.name, "class_weight": exp.class_weight}
        row.update(exp.oof)
        for metric in ("f1", "recall", "precisao", "roc_auc"):
            row[f"{metric}_folds_media"] = exp.folds[metric].mean()
            row[f"{metric}_folds_desvio"] = exp.folds[metric].std(ddof=1)
        for metric in ("acuracia", "precisao", "recall", "f1"):
            row[f"{metric}_limiar_05"] = exp.oof_05[metric]
        row["segundos_cv"] = exp.cv_seconds
        rows.append(row)
    table = pd.DataFrame(rows)
    # Criterio da regra 5: F1 OOF, depois recall, depois ROC/AUC.
    table = table.sort_values(["f1", "recall", "roc_auc"], ascending=False, kind="stable")
    table.insert(0, "posicao", range(1, len(table) + 1))
    table["escolhido"] = table["posicao"] == 1
    return table.reset_index(drop=True)


def run_cross_validation(train: pd.DataFrame) -> tuple[list[Experiment], pd.DataFrame]:
    X, y = split_features_target(train)
    groups = train[GROUP_COL]
    folds = make_folds(X, y, groups)
    rates = [y.iloc[valid].mean() for _, valid in folds]
    print(f"{N_FOLDS} folds sem cliente em treino e validacao do mesmo fold; "
          f"mau=1 por fold de {min(rates):.2%} a {max(rates):.2%}")
    experiments = build_experiments()
    assert len(experiments) == 7, f"Esperados 7 experimentos, obtidos {len(experiments)}"
    for exp in experiments:
        run_oof(exp, X, y, folds)
        print(f"  {exp.label:<36} limiar {exp.threshold:.4f} | F1 {exp.oof['f1']:.4f} | "
              f"recall {exp.oof['recall']:.4f} | AUC {exp.oof['roc_auc']:.4f} | "
              f"F1@0,5 {exp.oof_05['f1']:.4f} | {exp.cv_seconds:5.1f}s")
    return experiments, cv_table(experiments)


def pick_winner(experiments: list[Experiment], table: pd.DataFrame) -> Experiment:
    best = table.iloc[0]
    return next(e for e in experiments
                if e.name == best["experimento"] and e.class_weight == best["class_weight"])


# ------------------------------------------------- avaliacao no teste 2016


def evaluate_on_test(experiments: list[Experiment], table: pd.DataFrame, winner: Experiment,
                     train: pd.DataFrame, test: pd.DataFrame
                     ) -> tuple[pd.DataFrame, Pipeline, np.ndarray]:
    # O limiar de cada experimento vem do OOF; o teste so mede, nao escolhe nada.
    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)
    order = {(r.experimento, r.class_weight): r.posicao for r in table.itertuples()}
    rows, final_pipeline, final_proba = [], None, None
    for exp in experiments:
        pipeline = clone(exp.pipeline).fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        is_winner = exp is winner
        for kind, threshold in (("oof_f1", exp.threshold), ("fixo_0.5", 0.5)):
            row = {"posicao_oof": order[(exp.name, exp.class_weight)], "experimento": exp.name,
                   "class_weight": exp.class_weight, "limiar_tipo": kind}
            row.update(classification_metrics(y_test, proba, threshold))
            row["escolhido"] = is_winner
            rows.append(row)
        if is_winner:
            final_pipeline, final_proba = pipeline, proba
    metrics = pd.DataFrame(rows).sort_values("posicao_oof", kind="stable")
    return metrics.reset_index(drop=True), final_pipeline, final_proba


def plot_confusion(y: pd.Series, proba: np.ndarray, winner: Experiment) -> Path:
    pred = (proba >= winner.threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    share = cm / cm.sum()
    labels = np.array([[f"{n:,}\n({p:.1%})".replace(",", ".") for n, p in zip(rc, rp)]
                       for rc, rp in zip(cm, share)])
    blues = LinearSegmentedColormap.from_list("azul", ["#f0efec", COLOR_MAIN])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(cm, ax=ax, annot=labels, fmt="", cmap=blues, cbar=False, square=True,
                linewidths=2, linecolor=SURFACE, annot_kws={"fontsize": 11})
    ax.set_xticklabels(["bom (0)", "mau (1)"])
    ax.set_yticklabels(["bom (0)", "mau (1)"], rotation=0)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(f"Matriz de confusao no teste 2016\n{winner.label}, "
                 f"limiar OOF {winner.threshold:.4f}")
    return save_figure(fig, "matriz_confusao.png")


def plot_roc(y: pd.Series, proba: np.ndarray, winner: Experiment) -> Path:
    fpr, tpr, _ = roc_curve(y, proba)
    m = classification_metrics(y, proba, winner.threshold)
    point = (m["fp"] / (m["fp"] + m["tn"]), m["recall"])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, color=COLOR_MAIN, lw=2, label=f"{winner.label}, AUC {m['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], color=COLOR_REF, ls="--", lw=1, label="aleatorio")
    ax.plot(*point, "o", ms=9, color=COLOR_MAIN, mec=SURFACE, mew=2)
    ax.annotate(f"limiar OOF {winner.threshold:.4f}", point, xytext=(-12, 10), ha="right",
                textcoords="offset points", fontsize=9, color=COLOR_REF)
    ax.set_xlabel("taxa de falsos positivos")
    ax.set_ylabel("recall (taxa de verdadeiros positivos)")
    ax.set_title("Curva ROC no teste 2016")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal")
    return save_figure(fig, "roc.png")


def plot_precision_recall(y: pd.Series, proba: np.ndarray, winner: Experiment) -> Path:
    precision, recall, _ = precision_recall_curve(y, proba)
    m = classification_metrics(y, proba, winner.threshold)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    # Sem o ponto final convencional (precisao=1, recall=0), que nao vem de nenhum limiar.
    ax.plot(recall[:-1], precision[:-1], color=COLOR_MAIN, lw=2, label=winner.label)
    ax.axhline(y.mean(), color=COLOR_REF, ls="--", lw=1,
               label=f"taxa de mau no teste {y.mean():.2%}")
    ax.plot(m["recall"], m["precisao"], "o", ms=9, color=COLOR_MAIN, mec=SURFACE, mew=2)
    ax.annotate(f"limiar OOF {winner.threshold:.4f}\nF1 {m['f1']:.4f}",
                (m["recall"], m["precisao"]), xytext=(10, 6), textcoords="offset points",
                fontsize=9, color=COLOR_REF)
    ax.set_xlabel("recall")
    ax.set_ylabel("precisao")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precisao x recall no teste 2016")
    ax.legend(loc="upper right", frameon=False)
    return save_figure(fig, "precisao_recall.png")


# ------------------------------------------------- artefatos para o app


def save_model(pipeline: Pipeline) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 reduz o arquivo versionado sem alterar o modelo.
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    return MODEL_PATH


def rounded(metrics: dict, digits: int = 6) -> dict:
    def convert(value):
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        return round(float(value), digits)
    return {k: convert(v) for k, v in metrics.items()}


def build_metadata(winner: Experiment, test_metrics: pd.DataFrame,
                   train: pd.DataFrame, test: pd.DataFrame) -> dict:
    chosen = test_metrics[test_metrics["escolhido"]].set_index("limiar_tipo")
    return {
        "modelo": winner.name,
        "class_weight": winner.class_weight,
        "limiar": round(winner.threshold, 6),
        "regra_decisao": "mau previsto quando predict_proba[:, 1] >= limiar",
        "criterio_escolha": "maior F1 OOF; desempate por recall OOF e depois ROC/AUC OOF",
        "criterio_limiar": "limiar de maior F1 OOF (precision_recall_curve) no treino 2015",
        "parametros": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in winner.pipeline.named_steps["model"].get_params().items()},
        "metricas_oof": rounded(winner.oof),
        "metricas_oof_limiar_05": rounded(winner.oof_05),
        "metricas_teste": rounded(chosen.loc["oof_f1"].drop(
            ["posicao_oof", "experimento", "class_weight", "escolhido"]).to_dict()),
        "metricas_teste_limiar_05": rounded(chosen.loc["fixo_0.5"].drop(
            ["posicao_oof", "experimento", "class_weight", "escolhido"]).to_dict()),
        "features_numericas": NUMERIC_FEATURES,
        "features_categoricas": CATEGORICAL_FEATURES,
        "colunas_proibidas": FORBIDDEN_FEATURES,
        "alvo": TARGET,
        "split": {
            "treino": {"periodo": [train[DATE_COL].min(), train[DATE_COL].max()],
                       "linhas": int(len(train)), "taxa_mau": round(float(train[TARGET].mean()), 6)},
            "teste": {"periodo": [test[DATE_COL].min(), test[DATE_COL].max()],
                      "linhas": int(len(test)), "taxa_mau": round(float(test[TARGET].mean()), 6)},
        },
        "validacao_cruzada": {"tipo": "StratifiedGroupKFold", "n_splits": N_FOLDS,
                              "shuffle": True, "grupos": GROUP_COL},
        "random_state": RANDOM_STATE,
        "modelo_arquivo": MODEL_PATH.relative_to(MODEL_PATH.parents[1]).as_posix(),
        "versoes": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__,
                    "joblib": joblib.__version__},
    }


def decision_text(table: pd.DataFrame, winner: Experiment, test_metrics: pd.DataFrame) -> str:
    cols = ["posicao", "experimento", "class_weight", "limiar", "f1", "recall", "precisao",
            "roc_auc", "f1_limiar_05"]
    ranking = table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    chosen = test_metrics[test_metrics["escolhido"]].set_index("limiar_tipo")
    second = table.iloc[1]
    lines = [
        "Decisao do modelo",
        "=================",
        "",
        "Criterio (definido antes de olhar o teste): maior F1 out-of-fold no treino 2015,",
        f"StratifiedGroupKFold com {N_FOLDS} folds agrupados por {GROUP_COL}. "
        "Desempate por recall OOF e depois ROC/AUC OOF.",
        "Cada experimento e comparado no seu proprio limiar de maior F1 OOF.",
        "",
        "Ranking OOF (treino 2015):",
        ranking,
        "",
        f"Escolhido: {winner.label}, limiar {winner.threshold:.6f}.",
        f"F1 OOF {winner.oof['f1']:.4f} contra {second['f1']:.4f} do segundo colocado "
        f"({second['experimento']}, {second['class_weight']}).",
        "",
        "Teste 2016 (medido depois da escolha, nao usado para escolher modelo nem limiar):",
    ]
    for kind, label in (("oof_f1", f"limiar OOF {winner.threshold:.4f}"),
                        ("fixo_0.5", "limiar 0,5 (so comparacao)")):
        m = chosen.loc[kind]
        lines.append(f"  {label}: F1 {m['f1']:.4f} | recall {m['recall']:.4f} | "
                     f"precisao {m['precisao']:.4f} | acuracia {m['acuracia']:.4f} | "
                     f"AUC {m['roc_auc']:.4f} | TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}")
    return "\n".join(lines) + "\n"


def app_sample(train: pd.DataFrame) -> pd.DataFrame:
    sample = train.sample(n=N_APP_SAMPLE, random_state=RANDOM_STATE)
    return sample[[DATE_COL] + FEATURES + [TARGET]].sort_values(DATE_COL, kind="stable")


def form_options(train: pd.DataFrame) -> dict:
    # Opcoes do formulario so do treino: o app nao conhece o teste.
    numeric = {}
    for col in NUMERIC_FEATURES:
        values = train[col].dropna()
        is_int = bool(np.allclose(values, values.round()))
        cast = int if is_int else float
        numeric[col] = {"min": cast(values.min()), "max": cast(values.max()),
                        "mediana": cast(values.median()), "inteiro": is_int,
                        "permite_ausente": bool(train[col].isna().any())}
    categorical = {col: sorted(train[col].dropna().astype(str).unique().tolist())
                   for col in CATEGORICAL_FEATURES}
    return {"numericas": numeric, "categoricas": categorical}


def write_json(data: dict, name: str) -> Path:
    path = REPORTS_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_artifacts(table: pd.DataFrame, test_metrics: pd.DataFrame, winner: Experiment,
                   pipeline: Pipeline, proba: np.ndarray,
                   train: pd.DataFrame, test: pd.DataFrame) -> list[Path]:
    y_test = test[TARGET]
    paths = [REPORTS_DIR / "metricas_cv.csv", REPORTS_DIR / "metricas.csv",
             REPORTS_DIR / "limiares_teste.csv"]
    table.to_csv(paths[0], index=False, encoding="utf-8")
    test_metrics.to_csv(paths[1], index=False, encoding="utf-8")
    test_metrics[test_metrics["escolhido"]].to_csv(paths[2], index=False, encoding="utf-8")
    paths += [plot_confusion(y_test, proba, winner), plot_roc(y_test, proba, winner),
              plot_precision_recall(y_test, proba, winner), save_model(pipeline)]
    paths.append(write_json(build_metadata(winner, test_metrics, train, test), "metadata.json"))
    decision = REPORTS_DIR / "decisao_modelo.txt"
    decision.write_text(decision_text(table, winner, test_metrics), encoding="utf-8")
    sample = REPORTS_DIR / "dados_amostra.csv"
    app_sample(train).to_csv(sample, index=False, encoding="utf-8")
    paths += [decision, sample, write_json(form_options(train), "opcoes_formulario.json")]
    return paths


def check_artifacts(winner: Experiment, proba: np.ndarray, test: pd.DataFrame,
                    table: pd.DataFrame) -> None:
    X_test, _ = split_features_target(test)
    reloaded = joblib.load(MODEL_PATH)
    assert np.allclose(reloaded.predict_proba(X_test)[:, 1], proba), "joblib difere da memoria"
    metadata = json.loads((REPORTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    cv_csv = pd.read_csv(REPORTS_DIR / "metricas_cv.csv")
    assert np.isclose(metadata["limiar"], cv_csv.loc[cv_csv["escolhido"], "limiar"].iloc[0])
    assert metadata["modelo"] == table.iloc[0]["experimento"] == winner.name
    sample = pd.read_csv(REPORTS_DIR / "dados_amostra.csv")
    assert len(sample) == N_APP_SAMPLE and sample[DATE_COL].max() < TEST_START
    assert [p.name for p in MODELS_DIR.iterdir()] == [MODEL_PATH.name]
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"joblib recarregado: predict_proba no teste igual ao da memoria; "
          f"{MODEL_PATH.name} com {size_mb * 1000:.1f} KB (unico arquivo em models/)")
    print(f"Limiar do metadata {metadata['limiar']} igual ao do metricas_cv.csv; "
          f"dados_amostra.csv com {len(sample):,} linhas de "
          f"{sample[DATE_COL].min()} a {sample[DATE_COL].max()}")
    if size_mb > 100:
        print("ATENCAO: modelo acima de 100 MB, limite de arquivo do GitHub.")


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
    X_train, _ = split_features_target(train)
    X_test, _ = split_features_target(test)
    check_preprocessor(X_train, X_test)

    print("Validacao cruzada OOF no treino 2015:")
    experiments, table = run_cross_validation(train)
    winner = pick_winner(experiments, table)
    print(f"Escolhido pelo F1 OOF: {winner.label}, limiar {winner.threshold:.6f}")

    # So daqui em diante o teste 2016 e usado, e apenas para medir.
    test_metrics, pipeline, proba = evaluate_on_test(experiments, table, winner, train, test)
    print("Teste 2016 (limiar OOF de cada experimento e 0,5 para comparacao):")
    show = ["posicao_oof", "experimento", "class_weight", "limiar_tipo", "limiar", "acuracia",
            "precisao", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
    print(test_metrics[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for path in save_artifacts(table, test_metrics, winner, pipeline, proba, train, test):
        print(f"Salvo {path.relative_to(REPORTS_DIR.parent)}")
    check_artifacts(winner, proba, test, table)


if __name__ == "__main__":
    main()
