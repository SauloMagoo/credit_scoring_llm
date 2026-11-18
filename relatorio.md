# Relatorio - Credit Scoring com apoio de LLM

## 1. Introducao

O objetivo foi prever inadimplencia (`mau = 1`) com um pipeline sem vazamento de dados e
apresentar o resultado num app Streamlit didatico. A base tem 750.000 linhas, 16 colunas e
15 meses (2015-01 a 2016-03), com 7,82% de maus e 16.650 clientes que se repetem ao longo
dos meses. Todo o codigo foi gerado pelo Claude Code; o papel humano foi definir as regras
(CLAUDE.md), revisar cada diff, conferir os numeros e corrigir desvios.

## 2. Tratamento dos dados

- **Colunas**: `Unnamed: 0`, `index`, `data_ref` e o alvo ficaram fora das features; restaram
  5 numericas e 7 categoricas.
- **Ausentes**: so `tempo_emprego` (16,79%), imputado pela mediana com indicador de ausencia.
  Categoricas recebem moda e one-hot; numericas sao padronizadas. Tudo dentro de `Pipeline` +
  `ColumnTransformer`, ajustado so no treino de cada fold.
- **Duplicatas**: nenhuma exata, mas 469.661 linhas se repetem quando se ignora o numero da
  linha (o mesmo cliente varias vezes no mesmo mes). Foram mantidas; o risco de otimismo foi
  controlado agrupando a validacao por cliente.
- **Split temporal**: 50.000 linhas de 2015 para treino (mau 6,11%) e 15.000 de 2016 para teste
  (mau 14,67%), amostradas de forma estratificada por mes x alvo, filtrando o periodo antes.
- **Validacao**: `StratifiedGroupKFold` com 10 folds e grupos por `index`.
- **Desbalanceamento**: sem reamostragem; `class_weight` None e `"balanced"` foram testados como
  experimentos, e o limiar foi calibrado pelo F1 out-of-fold (OOF).

## 3. Resultados

Sete experimentos foram comparados pelo F1 OOF, cada um no seu limiar de maior F1:

| Experimento | F1 OOF | Recall | ROC/AUC |
|---|---|---|---|
| Arvore (None) | **0,2646** | 0,3397 | 0,7592 |
| Arvore (balanced) | 0,2618 | 0,3282 | 0,7580 |
| Regressao logistica (None) | 0,2446 | 0,4199 | 0,7584 |
| Regressao logistica (balanced) | 0,2398 | 0,4504 | 0,7557 |
| Random forest (None) | 0,2010 | 0,3868 | 0,6920 |
| Random forest (balanced) | 0,1955 | 0,4527 | 0,6941 |
| KNN | 0,1556 | 0,3953 | 0,6153 |

O vencedor foi a arvore de decisao com `max_depth=4` e `class_weight=None`, com limiar
0,1374. Avaliado uma unica vez no teste de 2016, depois da escolha:

| Limiar | Acuracia | Precisao | Recall | F1 | ROC/AUC |
|---|---|---|---|---|---|
| 0,1374 (OOF) | 0,6559 | 0,2544 | 0,6965 | 0,3726 | 0,7285 |
| 0,5 | 0,8524 | 0,1905 | 0,0018 | 0,0036 | 0,7285 |

Matriz de confusao no limiar OOF: 8.305 TN, 4.494 FP, 668 FN e 1.533 TP.

## 4. Discussao

- **O limiar importa mais que o modelo.** Com 0,5 a arvore acerta 4 de 2.201 maus no teste; com
  o limiar calibrado no OOF, 1.533. A acuracia de 0,8524 no limiar 0,5 fica abaixo de prever
  "bom" para todos (85,33% no teste, ou seja, 1 - 0,1467), o que mostra que acuracia nao serve
  de criterio aqui.
- **Peso de classe x limiar.** Nas arvores, `"balanced"` so deslocou as probabilidades (limiar
  0,7238 contra 0,1374) e chegou a quase o mesmo F1. Calibrar o limiar fez o mesmo papel que o
  peso de classe.
- **Modelos simples venceram.** A arvore rasa e a regressao logistica tiveram ROC/AUC OOF de
  ~0,76; a random forest sem limite de profundidade (0,69) e o KNN (0,62) ficaram atras. Uma
  explicacao provavel, nao testada, e sobreajuste as linhas repetidas, sem ajuste de
  hiperparametros.
- **Drift.** O F1 do teste (0,3726) e maior que o OOF (0,2646) porque ha mais maus em 2016, nao
  porque o modelo melhorou. O AUC caiu de 0,7592 para 0,7285, e o limiar calibrado com ~6% de
  maus gera muitos falsos positivos quando a taxa real e ~15%.
- **Limitacoes.** Cerca de 95% dos clientes do teste (9.421) tambem estao no treino; a diferenca
  entre as duas arvores (0,0028 de F1) e pequena demais para ser robusta; e foram usadas 65.000
  das 750.000 linhas. O modelo e didatico, nao serve para conceder credito.

## 5. Reflexao critica sobre o uso do Claude Code

**O que funcionou.** O agente produziu codigo organizado rapidamente, respeitou as regras do
CLAUDE.md (Pipeline, grupos por cliente, teste lido so depois da escolha) e incluiu asserts que
verificam essas regras durante a execucao (nenhum cliente em treino e validacao do mesmo fold,
nenhuma coluna proibida nas features, joblib recarregado igual ao modelo em memoria). Tambem
investigou por conta propria as 469.661 linhas repetidas. O pipeline completo roda em ~48 s, e
as metricas foram iguais, com 4 casas, em tres execucoes.

**O que exigiu correcao.** O agente errou ou passou do escopo em quatro pontos registrados no
diario: partiu de 8 experimentos em vez de 7; tentou criar `src/eda.py` fora da estrutura
pedida; gravou um arquivo de memoria fora do projeto sem pedir permissao; e propos usar linhas
do teste de 2016 como amostra do app, contrariando a regra de o app nao usar o teste. Nenhum
desses erros apareceria nas metricas: foram encontrados lendo diffs e comparando com o guia.
Houve ainda uma divergencia real com a referencia (a numerica de maior correlacao com o alvo
foi `tempo_emprego`, -0,13, nao `renda`), registrada em vez de ajustada.

**Licoes.** (1) Regras escritas e verificaveis (CLAUDE.md, asserts no codigo) rendem mais que
instrucoes soltas. (2) O agente tende a fazer mais do que foi pedido; revisar cada mudanca antes
de aprovar e o principal controle. (3) Numeros so devem ir para a documentacao a partir dos
artefatos gerados, nunca da memoria do agente: este relatorio usa apenas `reports/*.csv`,
`reports/metadata.json` e `reports/eda_fatos.json`. (4) As decisoes de metodologia (o que e
vazamento, qual metrica escolher) continuaram humanas; o agente acelerou a execucao, nao
substituiu o julgamento.
