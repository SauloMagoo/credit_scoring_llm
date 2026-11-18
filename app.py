"""App Streamlit didatico de credit scoring.

Apenas le os artefatos gerados por src/train_eval.py (reports/ e models/modelo.joblib).
Nao treina, nao ajusta limiar e nao le a base bruta nem o teste 2016.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "modelo.joblib"
TRAIN_CMD = "python src/train_eval.py"

REQUIRED = [
    MODEL_PATH,
    REPORTS / "metadata.json",
    REPORTS / "eda_fatos.json",
    REPORTS / "opcoes_formulario.json",
    REPORTS / "metricas.csv",
    REPORTS / "metricas_cv.csv",
    REPORTS / "limiares_teste.csv",
    REPORTS / "dados_amostra.csv",
]

st.set_page_config(page_title="Credit Scoring didatico", page_icon="💳", layout="wide")


# ------------------------------------------------------------------ leitura

@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if "class_weight" in df.columns:
        df = df.rename(columns={"class_weight": "peso_classes"})
    if "peso_classes" in df.columns:
        # O pandas le a string "None" como NaN; sem isso a arvore sem peso some das tabelas.
        # astype(object): se todas as linhas forem "None" a coluna vem como float.
        df["peso_classes"] = df["peso_classes"].astype(object).fillna("None").astype(str)
    return df


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def missing_artifacts() -> list[Path]:
    return [p for p in REQUIRED if not p.exists()]


def show_missing(paths: list[Path]) -> None:
    st.error("Artefatos ausentes. Gere-os antes de abrir o app:")
    st.code(TRAIN_CMD, language="bash")
    st.write("Arquivos nao encontrados:")
    for p in paths:
        st.write(f"- `{p.relative_to(ROOT)}`")


def show_image(name: str, caption: str | None = None) -> None:
    path = REPORTS / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"`reports/{name}` nao encontrado. Rode `{TRAIN_CMD}`.")


def pct(x: float, casas: int = 2) -> str:
    return f"{x * 100:.{casas}f}%".replace(".", ",")


def num(x: float, casas: int = 4) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def inteiro(x: int) -> str:
    return f"{int(x):,}".replace(",", ".")


# ------------------------------------------------------------------ sidebar

def sidebar(meta: dict) -> None:
    with st.sidebar:
        st.header("Modelo em uso")
        st.metric("Modelo", meta["modelo"])
        st.metric("Peso das classes", str(meta["class_weight"]))
        st.metric("Limiar calibrado", num(meta["limiar"]))
        st.caption(meta["regra_decisao"])
        st.caption(f"Limiar: {meta['criterio_limiar']}")
        st.subheader("Versoes")
        st.table(pd.DataFrame(meta["versoes"].items(), columns=["pacote", "versao"])
                 .set_index("pacote"))
        st.warning(
            "Uso exclusivamente didatico. O modelo tem desempenho modesto "
            f"(AUC no teste {num(meta['metricas_teste']['roc_auc'], 3)}) e nao deve "
            "ser usado para decisoes reais de credito."
        )


# ------------------------------------------------------------------ abas

def tab_dados(fatos: dict, meta: dict, amostra: pd.DataFrame) -> None:
    st.subheader("Base completa")
    c = st.columns(5)
    c[0].metric("Linhas", inteiro(fatos["linhas"]))
    c[1].metric("Clientes (index)", inteiro(fatos["n_clientes"]))
    c[2].metric("Meses", fatos["n_meses"])
    c[3].metric("Taxa de mau", pct(fatos["taxa_alvo"]))
    c[4].metric("tempo_emprego ausente", pct(fatos["ausentes"].get("tempo_emprego", 0)))

    tr, te = meta["split"]["treino"], meta["split"]["teste"]
    c = st.columns(4)
    c[0].metric("Treino 2015", inteiro(tr["linhas"]), help=" a ".join(tr["periodo"]))
    c[1].metric("Mau no treino", pct(tr["taxa_mau"]))
    c[2].metric("Teste 2016", inteiro(te["linhas"]), help=" a ".join(te["periodo"]))
    c[3].metric("Mau no teste", pct(te["taxa_mau"]))
    st.caption(
        f"Referencias ingenuas: acuracia de prever sempre 0 = {pct(fatos['baseline_acuracia_sempre_0'])}; "
        f"F1 de prever sempre 1 = {num(fatos['baseline_f1_sempre_1'])}. "
        f"Duplicatas exatas: {fatos['duplicatas_exatas']}."
    )

    st.subheader("Drift da taxa de inadimplencia")
    mn, mx = fatos["drift_taxa_mensal_min"], fatos["drift_taxa_mensal_max"]
    st.write(
        f"A taxa mensal de mau vai de **{pct(mn['taxa'])}** ({mn['mes'][:7]}) a "
        f"**{pct(mx['taxa'])}** ({mx['mes'][:7]}). Por isso o split e temporal: "
        "o teste de 2016 simula o uso do modelo no futuro."
    )
    show_image("eda_target_mes.png")

    st.subheader("Amostra do treino 2015")
    st.caption(f"{inteiro(len(amostra))} linhas aleatorias do treino (random_state=42). "
               "Nenhuma linha do teste 2016.")
    filtrada = filtrar_amostra(amostra)
    st.write(f"{inteiro(len(filtrada))} linhas apos filtros | taxa de mau "
             f"{pct(filtrada['mau'].mean()) if len(filtrada) else '-'}")
    st.dataframe(filtrada, width="stretch", height=320)

    st.subheader("Graficos da EDA (base completa)")
    e1, e2 = st.columns(2)
    with e1:
        show_image("eda_faltantes.png")
    with e2:
        show_image("eda_correlacao.png")
    show_image("eda_numericas.png")
    show_image("eda_categoricas.png")


def filtrar_amostra(df: pd.DataFrame) -> pd.DataFrame:
    c = st.columns(4)
    meses = sorted(df["data_ref"].unique())
    sel_meses = c[0].multiselect("Mes (data_ref)", meses)
    sel_mau = c[1].multiselect("mau", [0, 1])
    sel_renda = c[2].multiselect("tipo_renda", sorted(df["tipo_renda"].dropna().unique()))
    idade_min, idade_max = int(df["idade"].min()), int(df["idade"].max())
    faixa = c[3].slider("idade", idade_min, idade_max, (idade_min, idade_max))
    mask = df["idade"].between(*faixa)
    if sel_meses:
        mask &= df["data_ref"].isin(sel_meses)
    if sel_mau:
        mask &= df["mau"].isin(sel_mau)
    if sel_renda:
        mask &= df["tipo_renda"].isin(sel_renda)
    return df[mask]


def tab_tratamento(meta: dict) -> None:
    st.subheader("Decisoes de tratamento")
    st.markdown(
        f"""
Todo o pre-processamento esta dentro de um `Pipeline` com `ColumnTransformer`, ajustado
somente nos dados de treino (em cada fold da validacao cruzada e, no final, no treino 2015).

| Grupo | Colunas | Passos |
|---|---|---|
| Numericas | {", ".join(f"`{c}`" for c in meta["features_numericas"])} | `SimpleImputer(median, add_indicator=True)` + `StandardScaler` |
| Categoricas | {", ".join(f"`{c}`" for c in meta["features_categoricas"])} | `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")` |
| Descartadas | {", ".join(f"`{c}`" for c in meta["colunas_proibidas"])} | `remainder="drop"` |

- **Ausentes em `tempo_emprego`**: imputados pela mediana do treino, com uma coluna indicadora
  de ausencia (o fato de faltar pode ser informativo).
- **Outliers** (ex.: renda): mantidos; a arvore de decisao e pouco sensivel a escala.
- **Categorias novas** no teste ou no formulario: ignoradas pelo `OneHotEncoder`.
- **Linhas repetidas** do mesmo cliente: mantidas; o agrupamento por `index` na validacao
  impede que o mesmo cliente apareca em treino e validacao.
"""
    )

    st.subheader("Prevencao de vazamento")
    cv = meta["validacao_cruzada"]
    tr, te = meta["split"]["treino"], meta["split"]["teste"]
    st.markdown(
        f"""
1. **Colunas proibidas**: {", ".join(f"`{c}`" for c in meta["colunas_proibidas"])} nunca sao features.
2. **Split temporal**: treino {tr["periodo"][0][:7]} a {tr["periodo"][1][:7]} ({inteiro(tr["linhas"])} linhas)
   e teste {te["periodo"][0][:7]} a {te["periodo"][1][:7]} ({inteiro(te["linhas"])} linhas), amostrados
   de forma estratificada por mes e alvo. O periodo e filtrado antes de amostrar.
3. **Validacao cruzada**: `{cv["tipo"]}` com {cv["n_splits"]} folds e grupos por `{cv["grupos"]}`.
4. **Escolha do modelo**: {meta["criterio_escolha"]}. O teste 2016 nao participa.
5. **Limiar**: {meta["criterio_limiar"]}. O teste nunca e usado para isso.
6. **Teste 2016**: avaliado uma unica vez, depois da escolha.
7. **Este app**: apenas le `reports/` e `models/modelo.joblib`; nao retreina nada.
8. **Sementes**: `random_state={meta["random_state"]}` em amostragem, folds e modelos.
"""
    )


def tab_aprendizado(cv: pd.DataFrame) -> None:
    st.subheader("Comparacao dos experimentos (out-of-fold, treino 2015)")
    st.caption("Cada experimento e avaliado no seu proprio limiar de maior F1 OOF. "
               "Ordenado por F1, desempate por recall e ROC/AUC.")
    cv = cv.copy()
    cv["experimento_rotulo"] = cv["experimento"] + " (" + cv["peso_classes"] + ")"
    cols = ["posicao", "experimento", "peso_classes", "limiar", "f1", "recall",
            "precisao", "roc_auc", "acuracia", "f1_limiar_05", "escolhido"]
    st.dataframe(estilo(cv[cols]), width="stretch", hide_index=True)

    st.subheader("F1, recall e precisao OOF por experimento")
    grafico = cv.set_index("experimento_rotulo")[["f1", "recall", "precisao"]]
    st.bar_chart(grafico, stack=False, horizontal=True, height=380)

    st.subheader("Validacao cruzada: media e desvio entre os 10 folds")
    fold_cols = ["experimento", "peso_classes",
                 "f1_folds_media", "f1_folds_desvio",
                 "recall_folds_media", "recall_folds_desvio",
                 "precisao_folds_media", "precisao_folds_desvio",
                 "roc_auc_folds_media", "roc_auc_folds_desvio",
                 "tn", "fp", "fn", "tp", "segundos_cv"]
    st.dataframe(estilo(cv[fold_cols]), width="stretch", hide_index=True)


def estilo(df: pd.DataFrame):
    floats = {c: "{:.4f}" for c in df.select_dtypes("float").columns}
    return df.style.format(floats)


def tab_verificacao(meta: dict, limiares: pd.DataFrame, metricas: pd.DataFrame) -> None:
    st.subheader("Teste 2016 com o limiar calibrado")
    st.caption("Medido depois da escolha do modelo e do limiar; nao influenciou nenhuma decisao.")
    t, o = meta["metricas_teste"], meta["metricas_oof"]
    nomes = [("F1", "f1"), ("Recall", "recall"), ("Precisao", "precisao"),
             ("ROC/AUC", "roc_auc"), ("Acuracia", "acuracia")]
    c = st.columns(len(nomes))
    for col, (rotulo, k) in zip(c, nomes):
        col.metric(rotulo, num(t[k]), delta=f"{t[k] - o[k]:+.4f} vs OOF".replace(".", ","),
                   delta_color="off")
    t05 = meta["metricas_teste_limiar_05"]
    st.info(
        f"Com o limiar padrao 0,5 o recall no teste cairia para {pct(t05['recall'])} "
        f"(TP {t05['tp']} de {t05['tp'] + t05['fn']} maus). Por isso o limiar e calibrado."
    )

    st.markdown("**Matriz de confusao (teste, limiar calibrado)**")
    matriz = pd.DataFrame(
        [[t["tn"], t["fp"]], [t["fn"], t["tp"]]],
        index=["real 0 (bom)", "real 1 (mau)"],
        columns=["previsto 0", "previsto 1"],
    )
    m1, m2 = st.columns([1, 2])
    m1.dataframe(matriz, width="stretch")
    with m2:
        show_image("matriz_confusao.png")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Curva ROC**")
        show_image("roc.png")
    with g2:
        st.markdown("**Precisao x recall**")
        show_image("precisao_recall.png")

    st.subheader("Tabela de limiares (modelo escolhido, teste 2016)")
    cols = ["limiar_tipo", "limiar", "acuracia", "precisao", "recall", "f1", "roc_auc",
            "tn", "fp", "fn", "tp"]
    st.dataframe(estilo(limiares[cols]), width="stretch", hide_index=True)

    st.subheader("OOF (treino) x teste")
    comp = pd.DataFrame({
        "OOF limiar calibrado": meta["metricas_oof"],
        "Teste limiar calibrado": meta["metricas_teste"],
        "OOF limiar 0,5": meta["metricas_oof_limiar_05"],
        "Teste limiar 0,5": meta["metricas_teste_limiar_05"],
    })
    st.dataframe(comp, width="stretch")

    with st.expander("Todos os experimentos no teste 2016 (apenas informativo)"):
        st.caption("Nao usado para escolher modelo nem limiar.")
        st.dataframe(estilo(metricas), width="stretch", hide_index=True)


# ------------------------------------------------------------------ testar um caso

def defaults(opcoes: dict, amostra: pd.DataFrame) -> dict:
    valores = {}
    for col, info in opcoes["numericas"].items():
        mediana = info["mediana"]
        valores[col] = int(round(mediana)) if info["inteiro"] else float(round(mediana, 2))
    for col, cats in opcoes["categoricas"].items():
        moda = amostra[col].mode()
        valores[col] = moda.iloc[0] if len(moda) and moda.iloc[0] in cats else cats[0]
    valores["tempo_emprego_ausente"] = False
    return valores


def init_state(padrao: dict) -> None:
    for k, v in padrao.items():
        st.session_state.setdefault(f"in_{k}", v)
    st.session_state.setdefault("resultado", None)


def limpar(padrao: dict) -> None:
    for k, v in padrao.items():
        st.session_state[f"in_{k}"] = v
    st.session_state["resultado"] = None


ROTULOS = {
    "qtd_filhos": "Quantidade de filhos",
    "idade": "Idade",
    "tempo_emprego": "Tempo de emprego (anos)",
    "qt_pessoas_residencia": "Pessoas na residencia",
    "renda": "Renda",
    "sexo": "Sexo",
    "posse_de_veiculo": "Possui veiculo",
    "posse_de_imovel": "Possui imovel",
    "tipo_renda": "Tipo de renda",
    "educacao": "Educacao",
    "estado_civil": "Estado civil",
    "tipo_residencia": "Tipo de residencia",
}


def formulario(opcoes: dict) -> None:
    num_ops, cat_ops = opcoes["numericas"], opcoes["categoricas"]
    c1, c2, c3 = st.columns(3)
    with c1:
        for col in ("sexo", "estado_civil", "educacao", "tipo_renda"):
            st.selectbox(ROTULOS[col], cat_ops[col], key=f"in_{col}")
    with c2:
        for col in ("posse_de_veiculo", "posse_de_imovel"):
            st.selectbox(ROTULOS[col], cat_ops[col], key=f"in_{col}",
                         format_func=lambda v: {"S": "Sim", "N": "Nao"}.get(v, v))
        st.selectbox(ROTULOS["tipo_residencia"], cat_ops["tipo_residencia"],
                     key="in_tipo_residencia")
        for col in ("idade", "qtd_filhos", "qt_pessoas_residencia"):
            info = num_ops[col]
            st.slider(ROTULOS[col], int(info["min"]), int(info["max"]), key=f"in_{col}")
    with c3:
        r = num_ops["renda"]
        st.number_input(ROTULOS["renda"], min_value=float(r["min"]), max_value=float(r["max"]),
                        step=500.0, format="%.2f", key="in_renda",
                        help=f"Faixa observada: {r['min']:,.2f} a {r['max']:,.2f}")
        t = num_ops["tempo_emprego"]
        ausente = st.checkbox("Tempo de emprego desconhecido", key="in_tempo_emprego_ausente",
                              help="Envia o valor como ausente; o pipeline imputa a mediana do treino.")
        st.number_input(ROTULOS["tempo_emprego"], min_value=0.0,
                        max_value=float(round(t["max"], 2)), step=0.5, format="%.2f",
                        key="in_tempo_emprego", disabled=ausente)


def linha_formulario(meta: dict) -> pd.DataFrame:
    s = st.session_state
    linha = {col: s[f"in_{col}"] for col in meta["features_numericas"] + meta["features_categoricas"]}
    if s["in_tempo_emprego_ausente"]:
        linha["tempo_emprego"] = np.nan
    df = pd.DataFrame([linha], columns=meta["features_numericas"] + meta["features_categoricas"])
    for col in meta["features_numericas"]:
        df[col] = df[col].astype(float)
    return df


def tab_caso(modelo, meta: dict, opcoes: dict, amostra: pd.DataFrame) -> None:
    limiar = float(meta["limiar"])
    padrao = defaults(opcoes, amostra)
    init_state(padrao)

    st.subheader("Testar um caso")
    st.caption("Valores iniciais: medianas e categorias mais frequentes. "
               "Faixas limitadas ao observado na base.")
    formulario(opcoes)

    b1, b2, _ = st.columns([1, 1, 3])
    # st.button (e nao st.form_submit_button) para que Enter num campo nao dispare a analise.
    analisar = b1.button("Analisar crédito", type="primary", width="stretch")
    b2.button("Limpar campos e resultado", on_click=limpar, args=(padrao,),
              width="stretch")

    if analisar:
        linha = linha_formulario(meta)
        proba = float(modelo.predict_proba(linha)[0, 1])
        st.session_state["resultado"] = {"proba": proba, "linha": linha, "limiar": limiar}

    resultado = st.session_state.get("resultado")
    if resultado:
        mostrar_resultado(resultado)


def mostrar_resultado(res: dict) -> None:
    proba, limiar = res["proba"], res["limiar"]
    mau = proba >= limiar
    st.divider()
    c1, c2 = st.columns([1, 2])
    c1.metric("Probabilidade de inadimplencia", pct(proba))
    with c2:
        if mau:
            st.error(f"**Classificacao: MAU (alto risco)** — probabilidade {pct(proba)} "
                     f">= limiar {pct(limiar)}")
        else:
            st.success(f"**Classificacao: BOM (baixo risco)** — probabilidade {pct(proba)} "
                       f"< limiar {pct(limiar)}")
    st.progress(min(max(proba, 0.0), 1.0),
                text=f"Probabilidade {pct(proba)} | limiar calibrado {pct(limiar)}")
    st.caption("O limiar foi calibrado por F1 out-of-fold no treino 2015, por isso fica bem "
               "abaixo de 50%. A arvore tem profundidade 4, entao casos parecidos recebem a "
               "mesma probabilidade.")
    st.markdown("**Linha enviada ao modelo**")
    st.dataframe(res["linha"], width="stretch", hide_index=True)


# ------------------------------------------------------------------ main

def main() -> None:
    st.title("Credit Scoring com apoio de LLM")
    st.caption("Classificacao binaria de inadimplencia (mau = 1). App somente leitura.")

    ausentes = missing_artifacts()
    if ausentes:
        show_missing(ausentes)
        st.stop()

    meta = load_json(REPORTS / "metadata.json")
    fatos = load_json(REPORTS / "eda_fatos.json")
    opcoes = load_json(REPORTS / "opcoes_formulario.json")
    metricas = load_csv(REPORTS / "metricas.csv")
    cv = load_csv(REPORTS / "metricas_cv.csv")
    limiares = load_csv(REPORTS / "limiares_teste.csv")
    amostra = load_csv(REPORTS / "dados_amostra.csv")
    modelo = load_model(MODEL_PATH)

    sidebar(meta)
    abas = st.tabs(["Dados", "Tratamento", "Aprendizado", "Verificacao", "Testar um caso"])
    with abas[0]:
        tab_dados(fatos, meta, amostra)
    with abas[1]:
        tab_tratamento(meta)
    with abas[2]:
        tab_aprendizado(cv)
    with abas[3]:
        tab_verificacao(meta, limiares, metricas)
    with abas[4]:
        tab_caso(modelo, meta, opcoes, amostra)


main()
