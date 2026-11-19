# Credit Scoring com apoio de LLM

Pipeline de classificacao binaria de inadimplencia (`mau`) com split temporal, validacao
cruzada agrupada por cliente e limiar calibrado no treino, mais um app Streamlit didatico que
apenas le os artefatos gerados. Todo o codigo foi escrito com o Claude Code, sob revisao humana
registrada em [diario_de_bordo.md](diario_de_bordo.md). Relatorio curto: [relatorio.md](relatorio.md).

## Acesso rapido

- **App online**: https://credit-scoring-llm.streamlit.app/
- **Repositorio**: https://github.com/SauloMagoo/credit_scoring_llm
- **Diario de bordo**: [diario_de_bordo.md](diario_de_bordo.md)
- **Relatorio**: [relatorio.md](relatorio.md)
- **Analise critica do uso de IA**: [analise_critica_ia.md](analise_critica_ia.md)

Para rodar o app no proprio computador, siga as secoes "Instalacao" e "Execucao" abaixo.

## Base

- Arquivo: `data/credit_scoring.csv` (nao versionado; baixar do Kaggle e salvar nesse caminho).
- 750.000 linhas x 16 colunas, 15 meses (`data_ref` de 2015-01 a 2016-03).
- Alvo `mau` (booleano no CSV, convertido para 0/1): 58.643 maus (7,82%).
- 16.650 clientes distintos (`index`), repetidos em ate 15 meses.
- Unica coluna com ausentes: `tempo_emprego` (16,79%).
- 0 duplicatas exatas; 469.661 linhas repetidas quando se ignora `Unnamed: 0` (numero da linha).
- Taxa mensal de mau varia de 2,72% (2015-07) a 16,82% (2016-03): drift forte.

Features usadas (12):

| Tipo | Colunas |
|---|---|
| Numericas | qtd_filhos, idade, tempo_emprego, qt_pessoas_residencia, renda |
| Categoricas | sexo, posse_de_veiculo, posse_de_imovel, tipo_renda, educacao, estado_civil, tipo_residencia |

Nunca usadas como feature: `Unnamed: 0`, `index`, `data_ref` e o alvo `mau`.

## Instalacao (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Versoes registradas na ultima execucao: Python 3.13.17, scikit-learn 1.9.1, pandas 3.1.1,
numpy 2.5.4, joblib 1.6.0 (ver `requirements.txt` para streamlit, matplotlib e seaborn).

## Execucao

```powershell
# 1. Pipeline completo: EDA, split, CV, escolha, teste e artefatos (~48 s)
.\.venv\Scripts\python.exe src\train_eval.py

# 2. App (so le reports/ e models/modelo.joblib; nao precisa do CSV)
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O app tem cinco abas: Dados, Tratamento, Aprendizado, Verificacao e Testar um caso.

## Estrutura

```
CLAUDE.md              regras do projeto para o agente
PLANO.md               plano inicial da Etapa 0 (a estrutura final seguiu os prompts de cada etapa)
app.py                 app Streamlit (somente leitura)
src/config.py          caminhos, random_state, listas de features e colunas proibidas
src/train_eval.py      leitura, EDA, split, pre-processamento, CV, escolha, teste, artefatos
models/modelo.joblib   pipeline final (pre-processamento + arvore), unico modelo persistido
reports/
  eda_*.png, eda_fatos.json         analise exploratoria da base inteira
  metricas_cv.csv                   ranking OOF dos 7 experimentos (treino 2015)
  decisao_modelo.txt                criterio, ranking e escolha em texto
  metricas.csv                      os 7 experimentos no teste 2016 (limiar OOF e 0,5)
  limiares_teste.csv                modelo escolhido no teste, limiar OOF x 0,5
  matriz_confusao.png, roc.png, precisao_recall.png   graficos do teste 2016
  metadata.json                     modelo, limiar, parametros, metricas, split e versoes
  dados_amostra.csv                 5.000 linhas do treino 2015 para a aba Dados
  opcoes_formulario.json            faixas e categorias do treino para o formulario
diario_de_bordo.md     prompts, verificacoes, correcoes e decisoes humanas
relatorio.md           relatorio curto
```

## Decisoes principais

- **Sem vazamento**: todo pre-processamento esta em `Pipeline` + `ColumnTransformer`
  (`remainder="drop"`), ajustado so nos dados de treino de cada fold.
  - Numericas: `SimpleImputer(median, add_indicator=True)` + `StandardScaler`.
  - Categoricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
- **Split temporal**: treino = 50.000 linhas de 2015 (mau 6,11%); teste = 15.000 linhas de
  2016-01 a 2016-03 (mau 14,67%). O periodo e filtrado antes de amostrar; a amostragem e
  estratificada por mes x alvo com `random_state=42`.
- **Validacao**: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` com grupos
  por `index`; o codigo verifica que nenhum cliente aparece em treino e validacao do mesmo fold.
- **Experimentos (7)**: regressao logistica, arvore de decisao (`max_depth=4`) e random forest
  (300 arvores), cada um com `class_weight=None` e `"balanced"`, mais KNN (25 vizinhos,
  `weights="distance"`, sem `class_weight`).
- **Escolha**: maior F1 out-of-fold, cada experimento no seu proprio limiar de maior F1 OOF;
  desempate por recall e depois ROC/AUC. O teste 2016 so e lido depois da escolha.
- **Limiar**: o de maior F1 nas probabilidades OOF do vencedor (`precision_recall_curve`).
  Regra: mau previsto quando `predict_proba[:, 1] >= 0,137363`.
- **Duplicatas**: mantidas; a CV agrupada por cliente impede que linhas repetidas do mesmo
  cliente caiam em treino e validacao do mesmo fold.

## Resultados

Ranking OOF no treino 2015 (`reports/metricas_cv.csv`):

| # | Experimento | class_weight | Limiar | F1 | Recall | Precisao | ROC/AUC |
|---|---|---|---|---|---|---|---|
| 1 | arvore_decisao | None | 0,1374 | **0,2646** | 0,3397 | 0,2167 | 0,7592 |
| 2 | arvore_decisao | balanced | 0,7238 | 0,2618 | 0,3282 | 0,2177 | 0,7580 |
| 3 | regressao_logistica | None | 0,1128 | 0,2446 | 0,4199 | 0,1725 | 0,7584 |
| 4 | regressao_logistica | balanced | 0,6299 | 0,2398 | 0,4504 | 0,1634 | 0,7557 |
| 5 | random_forest | None | 0,1078 | 0,2010 | 0,3868 | 0,1357 | 0,6920 |
| 6 | random_forest | balanced | 0,1924 | 0,1955 | 0,4527 | 0,1247 | 0,6941 |
| 7 | knn | nao_aplicavel | 0,0773 | 0,1556 | 0,3953 | 0,0969 | 0,6153 |

Modelo escolhido: **arvore de decisao, `class_weight=None`, `max_depth=4`, limiar 0,1374**.
F1 por fold: media 0,2642, desvio 0,0264.

Modelo escolhido no teste 2016 (`reports/limiares_teste.csv`, medido depois da escolha):

| Limiar | Acuracia | Precisao | Recall | F1 | ROC/AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| 0,1374 (OOF) | 0,6559 | 0,2544 | 0,6965 | 0,3726 | 0,7285 | 8.305 | 4.494 | 668 | 1.533 |
| 0,5 (so comparacao) | 0,8524 | 0,1905 | 0,0018 | 0,0036 | 0,7285 | 12.782 | 17 | 2.197 | 4 |

Com o limiar padrao 0,5 o modelo quase nunca preve mau (4 de 2.201 maus no teste); a
calibracao do limiar no OOF e o que torna o modelo util. Os demais experimentos no teste
estao em `reports/metricas.csv`.

## Limitacoes

- **Drift**: a taxa de mau no teste (14,67%) e mais que o dobro da do treino (6,11%). O limiar
  foi calibrado com ~6% de maus, o que explica recall alto e precisao baixa no teste. O F1 do
  teste (0,3726) ser maior que o OOF (0,2646) reflete a prevalencia maior, nao um modelo melhor.
- **Desempenho modesto**: ROC/AUC 0,73 no teste; uso apenas didatico, nao para decisao de credito.
- **Clientes repetidos**: 9.421 clientes (cerca de 95% dos clientes do teste) aparecem tambem
  no treino, em meses diferentes. Nao e vazamento do alvo, mas deixa o teste parecido com o treino.
- **Linhas repetidas**: 469.661 linhas iguais ignorando `Unnamed: 0`; mantidas, pesam mais no treino.
- **Empate pratico**: as duas arvores diferem 0,0028 no F1 OOF; a escolha e sensivel a amostra.
- **Sem ajuste de hiperparametros**: parametros fixos e moderados; nao houve busca.
- **Amostra**: 65.000 das 750.000 linhas foram usadas na modelagem.
- **EDA**: a variavel numerica de maior correlacao (Pearson) com o alvo foi `tempo_emprego`
  (-0,13), nao `renda`, ao contrario da referencia do guia.
