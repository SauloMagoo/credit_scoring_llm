# Plano de implementacao - Credit Scoring

## Contexto
Repositorio ainda sem codigo: so a base `data/credit_scoring.csv` (750.000 x 16), `requirements.txt`,
`diario_de_bordo.md` e o `CLAUDE.md` com as regras. Objetivo: pipeline de classificacao de `mau`
(7,82% de positivos) sem vazamento, com split temporal 2015/2016, CV agrupada por cliente,
escolha por F1 out-of-fold, limiar calibrado no treino, e um app Streamlit que so le artefatos.
Cada etapa termina com verificacao rodada de verdade e registro no `diario_de_bordo.md`.

## Estrutura proposta
```
src/config.py      caminhos, RANDOM_STATE=42, listas de features, colunas proibidas
src/data.py        carga, conversao mau->int, amostragem temporal estratificada
src/features.py    build_preprocessor() -> ColumnTransformer
src/models.py      7 experimentos (LR/Arvore/RF x com/sem class_weight + KNN)
src/evaluate.py    OOF com StratifiedGroupKFold, metricas, busca de limiar
scripts/01_eda.py ... 05_avaliar_teste.py   (uma etapa por script)
app/app.py         Streamlit (somente leitura)
reports/           saidas (csv/json/png)       models/  modelo.joblib + metadata.json
```
Execucao sempre com `.\.venv\Scripts\python.exe <script>`. `data/processed/`, `models/` e
`reports/` tratados no `.gitignore` conforme decisao (dados processados nao versionados).

---

## Etapa 1 - Config, carga e EDA
- **Arquivos:** `src/config.py`, `src/data.py`, `scripts/01_eda.py`, `reports/eda_*.csv|png`.
- **Conteudo:** ler CSV (UTF-8), converter `mau` bool->int; conferir shape, taxa de mau, clientes
  unicos de `index`, % ausentes de `tempo_emprego`, duplicatas; taxa de mau e volume **por mes**;
  distribuicao das features por mes (drift: PSI 2015 vs 2016 por feature, sem uso em modelagem).
- **Verificacao:** imprimir shape (750000,16), taxa 7,82%, 16.650 clientes, 16,79% ausentes,
  0 duplicatas, intervalo de `data_ref` e meses disponiveis em 2015/2016.
- **Riscos de vazamento:** EDA e descritiva; nenhuma estatistica dela (medianas, categorias,
  cortes) pode ser hard-coded no pipeline. PSI so documenta o drift.

## Etapa 2 - Split temporal e amostragem
- **Arquivos:** `src/data.py` (funcao `sample_period`), `scripts/02_split.py`,
  `data/processed/treino_2015.csv`, `data/processed/teste_2016.csv`, `reports/split_resumo.json`.
- **Conteudo:** filtrar 2015 e 2016 por `data_ref`; amostrar 50.000 (2015) e 15.000 (2016)
  estratificando por estrato `mes x mau` (`train_test_split(..., stratify=estrato, random_state=42)`).
  As colunas `index` e `data_ref` sao mantidas no arquivo so para grupos/auditoria.
- **Verificacao:** contagens exatas 50.000/15.000; taxa de mau por mes na amostra ~= populacao
  (diferenca < 0,1 p.p.); anos sem sobreposicao; nº de clientes no treino; nº de clientes do teste
  que tambem aparecem no treino (registrar como limitacao, nao e vazamento de alvo mas infla
  a semelhanca).
- **Riscos:** amostrar antes de filtrar ano; usar 2016 em qualquer estatistica; esquecer que
  linhas do mesmo cliente caem no treino varias vezes (tratado pelos grupos da Etapa 4).

## Etapa 3 - Pre-processamento
- **Arquivos:** `src/features.py`.
- **Conteudo:** `ColumnTransformer`:
  - numericas (`qtd_filhos, idade, tempo_emprego, qt_pessoas_residencia, renda`):
    `SimpleImputer(strategy="median", add_indicator=True)` (indicador de ausencia de
    `tempo_emprego`, que tende a ser informativo) + `StandardScaler` (inofensivo para arvores);
  - categoricas (`sexo, posse_de_veiculo, posse_de_imovel, tipo_renda, educacao, estado_civil,
    tipo_residencia`): `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`;
  - `remainder="drop"` e selecao explicita de colunas -> `index`, `data_ref`, `Unnamed: 0`, `mau`
    nunca entram.
- **Verificacao:** script/teste que ajusta no treino e checa `get_feature_names_out()` sem nenhuma
  coluna proibida; `assert` que o X passado ao pipeline nao contem colunas proibidas.
- **Riscos:** imputar/escalar fora do Pipeline (mediana com dados de validacao/teste); categoria nova
  em 2016 quebrando o encoder (resolvido por `handle_unknown`).

## Etapa 4 - Comparacao de modelos com CV agrupada
- **Arquivos:** `src/models.py`, `src/evaluate.py`, `scripts/03_cv.py`,
  `reports/cv_resultados.csv`, `reports/oof_predicoes.csv`.
- **Experimentos (7):** LogisticRegression, DecisionTree e RandomForest com `class_weight=None`
  e `"balanced"` (6) + KNN (1), com `class_weight` registrado como `"nao_aplicavel"`.
  `random_state=42` onde existir. Hiperparametros fixos e moderados (arvore com
  `max_depth`/`min_samples_leaf`, RF com ~300 arvores e `n_jobs=-1`, KNN com `n_neighbors`
  relativamente alto, ex. 50, para gerar probabilidades menos discretas) - sem tuning extenso.
  O `StandardScaler` da Etapa 3 e essencial para o KNN; o KNN e o mais lento na predicao
  (50k x 10 folds), por isso o tempo por experimento sera medido.
- **CV:** `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`, `groups=index`;
  `cross_val_predict(method="predict_proba")` sobre o Pipeline completo -> probabilidades OOF.
  Metricas OOF: F1, recall, precisao, ROC/AUC, PR/AUC; tambem media/desvio por fold.
- **Escolha (decidido):** para cada experimento, F1 OOF medido no limiar que maximiza o seu
  proprio F1 OOF (recall e precisao reportados nesse limiar). Vence o maior F1; desempate por
  recall, depois ROC/AUC. O limiar do vencedor e o limiar final. Teste 2016 nao e lido aqui.
- **Verificacao:** assert de que nenhum `index` aparece em treino e validacao do mesmo fold;
  taxa de mau por fold ~7,8%; tabela ordenada impressa; tempo por modelo.
- **Riscos:** `KFold`/`StratifiedKFold` sem grupos (cliente em treino e validacao -> F1 otimista);
  pre-processar fora do Pipeline; olhar o teste "so para conferir" antes de escolher.

## Etapa 5 - Limiar e treino final
- **Arquivos:** `scripts/04_treino_final.py`, `models/modelo.joblib`, `models/metadata.json`,
  `reports/curva_limiar.png`.
- **Conteudo:** reutilizar o limiar de maior F1 OOF do vencedor (calculado na Etapa 4 com
  `precision_recall_curve` sobre as probabilidades OOF) e plotar F1/precisao/recall x limiar. Reajustar o Pipeline vencedor em todo o
  treino 2015 e salvar so ele com `joblib.dump`. Limiar, nome do modelo, metricas OOF, versoes de
  libs e lista de features vao para `metadata.json` (nao e outro modelo, so metadado).
- **Verificacao:** recarregar o joblib, prever o treino e checar shape/probabilidades em [0,1];
  limiar e F1 OOF impressos e iguais aos do relatorio.
- **Riscos:** calibrar limiar com previsoes in-sample (treino ajustado em tudo) ou com o teste.

## Etapa 6 - Avaliacao unica no teste 2016
- **Arquivos:** `scripts/05_avaliar_teste.py`, `reports/teste_2016.json`, `reports/teste_*.png`.
- **Conteudo:** carregar `modelo.joblib` + limiar e avaliar uma vez: F1, recall, precisao,
  ROC/AUC, PR/AUC, matriz de confusao, metricas por mes de 2016 e comparacao OOF x teste
  (quantifica o efeito do drift). Opcional: comparar com o limiar 0,5 apenas como informacao.
- **Verificacao:** 15.000 linhas avaliadas; numeros gravados no JSON e copiados ao diario.
- **Riscos:** voltar e trocar modelo/limiar por causa do teste (proibido; registrar como limitacao).

## Etapa 7 - App Streamlit
- **Arquivos:** `app/app.py`.
- **Conteudo:** paginas didaticas: base e drift (lendo `reports/`), metodologia anti-vazamento,
  resultados CV e teste, simulador de cliente (formulario -> `predict_proba` com o pipeline
  carregado via `st.cache_resource` e limiar do metadata).
- **Verificacao:** `.\.venv\Scripts\python.exe -m streamlit run app/app.py`; checar que o app
  nao importa `fit`/`cross_val*` (grep) e funciona sem o CSV bruto.
- **Riscos:** app retreinar ou ler o teste para recalcular metricas diferentes das registradas.

## Etapa 8 - Documentacao e diario
- Atualizar `diario_de_bordo.md` a cada etapa (prompt, verificacoes numericas reais, decisoes,
  correcoes, limitacoes: drift, clientes do teste presentes no treino, amostra de 65k de 750k).
- `README.md` curto com ordem de execucao dos scripts.

## Verificacao ponta a ponta
```
.\.venv\Scripts\python.exe scripts\01_eda.py
.\.venv\Scripts\python.exe scripts\02_split.py
.\.venv\Scripts\python.exe scripts\03_cv.py
.\.venv\Scripts\python.exe scripts\04_treino_final.py
.\.venv\Scripts\python.exe scripts\05_avaliar_teste.py
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```
Rodar duas vezes e confirmar resultados identicos (random_state=42).
