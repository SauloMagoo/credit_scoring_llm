# Diario de bordo

## Registro de interacoes

| # | Data/hora | Etapa | Prompt usado | Resposta resumida | O que foi conferido | Correcoes | Decisao humana | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 18/11 15:40 | Inicio | /init (sequencia da secao 6 do guia) | Listou os arquivos, leu requirements.txt, .gitignore, o diario e 3 linhas do CSV; propos reescrever o CLAUDE.md (cabecalho, estado do ambiente, secao sobre o diario, detalhes da base) | Diff comparado com o CLAUDE.md da secao 4 do guia | - | Edicao recusada: CLAUDE.md mantido igual ao guia ("substitua ou complemente com as regras acima") | OK |
| 2 | 18/11 15:44 | Etapa 0 | Prompt da Etapa 0 do guia, em modo de planejamento | Perguntou em qual limiar comparar o F1 OOF e quais seriam os 4 algoritmos, partindo de "8 candidatos"; apresentou um plano em etapas | Plano conferido com os 4 itens do guia: split temporal e CV por cliente, teste reservado, limiar por F1 OOF, acuracia nao e criterio unico | Premissa de 8 candidatos corrigida: 4 algoritmos do guia, KNN sem class_weight, 7 experimentos | Limiar: F1 OOF no limiar calibrado de cada experimento (Etapa 5 do guia). Plano aprovado pelos criterios do guia; a implementacao segue os prompts das Etapas 1 a 8 | OK |
| 3 | 18/11 15:53 | Etapa 1 | Prompt da Etapa 1 do guia | Criou src/config.py e src/train_eval.py; rodou o script; investigou por conta propria as duplicatas ignorando Unnamed: 0 (469.661) e encontrou o mesmo cliente repetido no mesmo mes (media 3,16, maximo 14) | Shape, taxa de mau, clientes, ausentes e colunas fora das features iguais ao guia; diffs revisados antes de aprovar | - | Manter todas as linhas (ver decisao 6) | OK |
| 4 | 18/11 16:03 | Etapa 2 | Prompt da Etapa 2 do guia | Tentou criar src/eda.py (recusado) e gravar memoria fora do projeto (recusado; apagou o arquivo que tinha criado). Depois colocou a EDA em src/train_eval.py, gerou os 5 PNGs e reports/eda_fatos.json e corrigiu rotulos do grafico de ausentes | Drift 2,72% (2015-07) a 16,82% (2016-03) e so tempo_emprego com ausentes, iguais ao guia. Divergencia: na correlacao de Pearson a mais forte foi tempo_emprego (-0,13), nao renda | Correcoes 2 e 3 | Versionar reports/ (artefatos da estrutura do guia); manter eda_fatos.json | OK |
| 5 | 18/11 16:08 | Etapa 3 | Prompt da Etapa 3 do guia | Criou o split temporal (filtra o periodo antes de amostrar), a amostragem estratificada por data_ref + mau, o ColumnTransformer e o build_pipeline; checou que o pre-processamento e ajustado so no treino | 50.000/15.000 linhas, mau 6,11%/14,67% e 9.421 clientes nos dois periodos, iguais ao guia; fit so no treino | - | Prosseguir | OK |
| 6 | 18/11 16:12 | Etapa 4 | Prompt da Etapa 4 do guia | Criou os 7 experimentos (LR, Arvore e RF com None e balanced; KNN nao_aplicavel), todos com build_pipeline, ajustados so no treino e mantidos em memoria | 7 experimentos; parametros iguais ao prompt; teste nao usado; nenhum modelo em disco (models/ inexistente); ajuste total ~3 s | - | Prosseguir | OK |
| 7 | 18/11 16:46 | Etapa 5 | Prompt da Etapa 5 do guia | Apresentou plano; implementou CV StratifiedGroupKFold (10 folds, grupos por index), limiar por F1 OOF, escolha por F1 OOF, avaliacao no teste e artefatos. Escolhido: arvore_decisao (class_weight=None), limiar 0,1374. Rodou o pipeline 3 vezes | Limiar, matriz e metricas no teste iguais a referencia do guia; teste lido so depois da escolha; folds sem cliente repetido; modelo.joblib 3,7 KB | Correcao 4 | Recusada a flag --sem-eda (fora do guia); models/modelo.joblib versionado (estrutura do guia) | OK |

## Decisoes humanas com pros e contras
1. Projeto em C:\Projetos (disco local). Pros: fora da sincronizacao do OneDrive. Contras: nenhum para o guia, que nao fixa o local.
2. .gitignore com .venv/, __pycache__/ e data/credit_scoring.csv. Pros: ambiente e base fora dos commits. Contras: um arquivo a mais que o guia nao traz.
3. Base baixada pelo navegador (opcao aceita pelo guia). Pros: sem instalar a CLI do Kaggle nem guardar token. Contras: passo manual.
4. /init usado (secao 6 do guia) e edicao do CLAUDE.md recusada. Pros: CLAUDE.md igual ao do guia. Contras: nenhum.
5. Plano da Etapa 0 aprovado pelos 4 criterios do guia, mesmo com estrutura e parametros diferentes das Etapas 1 a 8. Pros: segue o criterio do guia. Contras: PLANO.md difere da estrutura final; prevalecem os prompts de cada etapa.
6. Duplicatas: 0 exatas; 469.661 linhas repetidas ignorando Unnamed: 0 (numero da linha). Decisao: manter. Pros: preserva a base usada pelo guia na amostragem e nas metricas; a CV agrupada por index impede o mesmo cliente em treino e validacao do mesmo fold. Contras: linhas repetidas pesam mais no treino.
7. reports/ versionado, porque a estrutura final do guia inclui os artefatos de reports/. reports/eda_fatos.json (criado pelo agente, alem do pedido) mantido: nao altera numeros.
8. Recusada a flag --sem-eda proposta pelo agente: nao esta no guia.
9. models/modelo.joblib versionado: faz parte da estrutura final do guia (3,7 KB).
10. Interrompida uma quarta execucao do pipeline para comparar hashes (fora do guia), por causa do prazo; as metricas foram iguais nas tres rodadas, com 4 casas.

## Correcoes feitas depois de erros do agente
1. Etapa 0 (18/11 15:45): o agente partiu de 8 candidatos (4 algoritmos x 2 pesos de classe). Detectado comparando a pergunta com a secao 1 e a Etapa 4 do guia (7 experimentos, KNN "nao_aplicavel"). Corrigido na resposta ao agente.
2. Etapa 2 (18/11 16:00): o agente propos criar src/eda.py. Detectado comparando com a estrutura final da secao 1 do guia (codigo so em src/config.py e src/train_eval.py). Recusado; pedido para colocar a EDA em src/train_eval.py.
3. Etapa 2 (18/11 16:00): o agente gravou um arquivo de memoria fora do projeto sem pedir permissao e pediu para indexa-lo. Detectado pela mensagem "Wrote 1 memory" na tela. Recusado; o agente apagou o arquivo. As regras ficam so no CLAUDE.md, como o guia define.
4. Etapa 5 (18/11 16:16): o agente propos gerar reports/dados_amostra.csv com 5 linhas do teste 2016. Detectado comparando com a Etapa 6 do guia ("o app nao usa o teste") e com a aba Dados ("amostra filtravel"). Corrigido: 5.000 linhas aleatorias do treino 2015 (random_state=42).

## Verificacoes numericas (shape, taxa, clientes, ausentes, metricas)
| Item | Referencia do guia | Obtido |
|---|---|---|
| Dimensoes | 750.000 x 16 | (750000, 16) |
| mau = 1 | 58.643 (7,82%) | 58.643 |
| mau = 0 | 691.357 | 691.357 |
| Clientes distintos (index) | 16.650 | 16.650 |
| tempo_emprego ausente | 16,79% | 16,79% |
| Duplicatas exatas | 0 | 0 |
| Duplicatas ignorando Unnamed: 0 | (nao informado) | 469.661 |
| Drift mensal | 2,72% a 16,82% | 2,72% (2015-07) a 16,82% (2016-03) |
| Baseline (acuracia, base inteira) | 92,18% | 92,18% |
| Ausentes relevantes | so tempo_emprego | so tempo_emprego (16,79%) |
| Variavel com maior sinal | renda | tempo_emprego (Pearson -0,13); divergente |
| Treino / teste | 50.000 / 15.000 | 50.000 / 15.000 |
| mau=1 treino / teste | 6,11% / 14,67% | 6,11% / 14,67% |
| Clientes nos dois periodos (amostras) | ~9.421 | 9.421 |
| Limiar calibrado | 0,137 | 0,1374 |
| Recall com limiar 0,5 | 0,18% | 0,18% |
| Matriz (TN/FP/FN/TP) | 8.305 / 4.494 / 668 / 1.533 | 8.305 / 4.494 / 668 / 1.533 |
| Acuracia / precisao / recall / F1 / AUC (teste) | 0,656 / 0,254 / 0,697 / 0,373 / 0,729 | 0,6559 / 0,2544 / 0,6965 / 0,3726 / 0,7285 |
| Tempo do pipeline completo | ~9-10 min | ~48 s |

## Limitacoes conhecidas

- O mesmo cliente aparece varias vezes no mesmo mes (469.661 linhas repetidas ignorando Unnamed: 0).
- Drift forte: taxa de mau de 2,72% (2015-07) a 16,82% (2016-03); o teste (2016) tem taxa bem acima do treino.
- Na EDA, a correlacao de Pearson (renda bruta) nao mostrou a renda como variavel de maior sinal, ao contrario do guia.
- 95% dos clientes do teste tambem estao no treino (9.421); nao e vazamento do alvo (meses diferentes), mas deixa o teste parecido com o treino.
- Precisao baixa no teste (0,254): o limiar foi calibrado com ~6% de maus no treino e o teste tem ~15%.
- AUC ~0,73: desempenho modesto; uso apenas didatico.
- As duas arvores (None e balanced) quase empatam no F1 OOF (diferenca de 0,0028).
- Entre execucoes, alguns arquivos de reports/ mudaram (coluna de tempo em metricas_cv.csv; uma diferenca sem causa identificada entre a 2a e a 3a rodada); as metricas foram iguais com 4 casas.