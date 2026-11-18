# Diario de bordo

## Registro de interacoes

| # | Data/hora | Etapa | Prompt usado | Resposta resumida | O que foi conferido | Correcoes | Decisao humana | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 18/11 15:40 | Inicio | /init (sequencia da secao 6 do guia) | Listou os arquivos, leu requirements.txt, .gitignore, o diario e 3 linhas do CSV; propos reescrever o CLAUDE.md (cabecalho, estado do ambiente, secao sobre o diario, detalhes da base) | Diff comparado com o CLAUDE.md da secao 4 do guia | - | Edicao recusada: CLAUDE.md mantido igual ao guia ("substitua ou complemente com as regras acima") | OK |
| 2 | 18/11 15:44 | Etapa 0 | Prompt da Etapa 0 do guia, em modo de planejamento | Perguntou em qual limiar comparar o F1 OOF e quais seriam os 4 algoritmos, partindo de "8 candidatos"; apresentou um plano em etapas | Plano conferido com os 4 itens do guia: split temporal e CV por cliente, teste reservado, limiar por F1 OOF, acuracia nao e criterio unico | Premissa de 8 candidatos corrigida: 4 algoritmos do guia, KNN sem class_weight, 7 experimentos | Limiar: F1 OOF no limiar calibrado de cada experimento (Etapa 5 do guia). Plano aprovado pelos criterios do guia; a implementacao segue os prompts das Etapas 1 a 8 | OK |
| 3 | 18/11 15:53 | Etapa 1 | Prompt da Etapa 1 do guia | Criou src/config.py e src/train_eval.py; rodou o script; investigou por conta propria as duplicatas ignorando Unnamed: 0 (469.661) e encontrou o mesmo cliente repetido no mesmo mes (media 3,16, maximo 14) | Shape, taxa de mau, clientes, ausentes e colunas fora das features iguais ao guia; diffs revisados antes de aprovar | - | Manter todas as linhas (ver decisao 6) | OK |
| 4 | 18/11 16:03 | Etapa 2 | Prompt da Etapa 2 do guia | Tentou criar src/eda.py (recusado) e gravar memoria fora do projeto (recusado; apagou o arquivo que tinha criado). Depois colocou a EDA em src/train_eval.py, gerou os 5 PNGs e reports/eda_fatos.json e corrigiu rotulos do grafico de ausentes | Drift 2,72% (2015-07) a 16,82% (2016-03) e so tempo_emprego com ausentes, iguais ao guia. Divergencia: na correlacao de Pearson a mais forte foi tempo_emprego (-0,13), nao renda | Correcoes 2 e 3 | Versionar reports/ (artefatos da estrutura do guia); manter eda_fatos.json | OK |

## Decisoes humanas com pros e contras
1. Projeto em C:\Projetos (disco local). Pros: fora da sincronizacao do OneDrive. Contras: nenhum para o guia, que nao fixa o local.
2. .gitignore com .venv/, __pycache__/ e data/credit_scoring.csv. Pros: ambiente e base fora dos commits. Contras: um arquivo a mais que o guia nao traz.
3. Base baixada pelo navegador (opcao aceita pelo guia). Pros: sem instalar a CLI do Kaggle nem guardar token. Contras: passo manual.
4. /init usado (secao 6 do guia) e edicao do CLAUDE.md recusada. Pros: CLAUDE.md igual ao do guia. Contras: nenhum.
5. Plano da Etapa 0 aprovado pelos 4 criterios do guia, mesmo com estrutura e parametros diferentes das Etapas 1 a 8. Pros: segue o criterio do guia. Contras: PLANO.md difere da estrutura final; prevalecem os prompts de cada etapa.
6. Duplicatas: 0 exatas; 469.661 linhas repetidas ignorando Unnamed: 0 (numero da linha). Decisao: manter. Pros: preserva a base usada pelo guia na amostragem e nas metricas; a CV agrupada por index impede o mesmo cliente em treino e validacao do mesmo fold. Contras: linhas repetidas pesam mais no treino.
7. reports/ versionado, porque a estrutura final do guia inclui os artefatos de reports/. reports/eda_fatos.json (criado pelo agente, alem do pedido) mantido: nao altera numeros.

## Correcoes feitas depois de erros do agente
1. Etapa 0 (18/11 15:45): o agente partiu de 8 candidatos (4 algoritmos x 2 pesos de classe). Detectado comparando a pergunta com a secao 1 e a Etapa 4 do guia (7 experimentos, KNN "nao_aplicavel"). Corrigido na resposta ao agente.
2. Etapa 2 (18/11 16:00): o agente propos criar src/eda.py. Detectado comparando com a estrutura final da secao 1 do guia (codigo so em src/config.py e src/train_eval.py). Recusado; pedido para colocar a EDA em src/train_eval.py.
3. Etapa 2 (18/11 16:00): o agente gravou um arquivo de memoria fora do projeto sem pedir permissao e pediu para indexa-lo. Detectado pela mensagem "Wrote 1 memory" na tela. Recusado; o agente apagou o arquivo. As regras ficam so no CLAUDE.md, como o guia define.

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

## Limitacoes conhecidas

- O mesmo cliente aparece varias vezes no mesmo mes (469.661 linhas repetidas ignorando Unnamed: 0).
- Drift forte: taxa de mau de 2,72% (2015-07) a 16,82% (2016-03); o teste (2016) tem taxa bem acima do treino.
- Na EDA, a correlacao de Pearson (renda bruta) nao mostrou a renda como variavel de maior sinal, ao contrario do guia.