# CLAUDE.md - Credit Scoring com apoio de LLM

## Objetivo
Construir um pipeline completo de classificacao binaria de inadimplencia e um app
Streamlit didatico, sem vazamento de dados.

## Base
- CSV: data/credit_scoring.csv (750.000 x 16).
- Alvo: mau (1 = inadimplente, 0 = bom pagador). No CSV original vem como booleano.
- Identificadores/data que NAO entram como features: Unnamed: 0, index, data_ref.
- index identifica 16.650 clientes repetidos em ate 15 meses.
- tempo_emprego tem 16,79% de ausentes.
- Nao ha duplicatas exatas.

## Regras obrigatorias
1. Todo pre-processamento deve estar dentro de Pipeline + ColumnTransformer e ser ajustado
   somente no treino.
2. Nunca usar index, data_ref, Unnamed: 0 ou o alvo como feature.
3. Split temporal: treino em 2015 (50.000 linhas) e teste em 2016 (15.000 linhas),
   amostrados de forma estratificada por mes e alvo.
4. Validacao cruzada StratifiedGroupKFold de 10 folds, com grupos por index.
5. Escolher o modelo pelo maior F1 out-of-fold. Desempate por recall e depois ROC/AUC.
   O teste de 2016 nao pode ser usado na escolha.
6. Calibrar o limiar de decisao por F1 out-of-fold no treino. Nunca usar o teste para isso.
7. Persistir apenas o pipeline final em models/modelo.joblib com joblib.
8. O app Streamlit deve apenas ler os artefatos; nao deve retreinar.
9. Usar random_state=42 em amostragem, split, modelos e folds.
10. Nao inventar metricas. Rodar o codigo e registrar os numeros reais.

## Estilo
- Python 3.10+, type hints quando ajudarem, funcoes pequenas e nomes claros.
- Comentarios apenas quando explicarem uma decisao nao obvia.
- Encoding UTF-8.
- PowerShell no Windows: nao usar &&; usar ; ou comandos separados.
- Antes de grandes mudancas, apresentar um plano curto e esperar aprovacao.
- Depois de cada etapa, rodar uma verificacao e mostrar a saida relevante.
- Nao executar comandos destrutivos de git sem pedido explicito.