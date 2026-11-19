# Analise critica do uso de IA no projeto

Este documento registra como a IA foi usada no projeto, onde ela errou, como cada erro foi percebido
e o que foi aprendido. Complementa o diario_de_bordo.md (registro etapa por etapa) e a secao de
reflexao do relatorio.md.

## 1. Como a IA foi usada

- **Guia passo a passo**: o projeto seguiu um guia passo a passo, elaborado com apoio de IA e
  revisado por 1 humano antes de qualquer validacao final, que descreve as etapas e os prompts para
  que qualquer pessoa consiga reproduzir o app com o Claude Code.
- **Claude Code** (agente de programacao): escreveu todo o codigo, a partir dos prompts do guia,
  sempre em modo manual (cada edicao e cada comando aprovados antes de executar).
- **Assistente de IA em chat**: usado para explicar as etapas, revisar as saidas e os diffs e
  rascunhar respostas ao agente.

## 2. Primeira tentativa: descartada por falta de controle

O projeto foi comecado uma primeira vez e abandonado na Etapa 0. Os problemas vieram
principalmente do assistente em chat, nao do Claude Code:

| Problema | O que aconteceu |
|---|---|
| Acoes sem pedido | Criou e atualizou um documento de acompanhamento sem ser pedido, avisando so depois |
| Registros falsos | Disse que tinha guardado um texto que nao guardou e registrou como "regra do usuario" uma frase que o usuario nunca disse |
| Decisoes dele apresentadas como humanas | Escolhas do assistente foram registradas como "decisao humana" no diario |
| Desvio do material de referencia | Ao revisar o plano do agente, acrescentou detalhes de um codigo de referencia e preferencias proprias que nao tinham sido pedidos |
| Afirmacoes sem conferir | Afirmou fatos e numeros que nao tinha verificado e seguiu sem ver o resultado de testes que ele mesmo pediu |
| Material de base inadequado | A primeira tentativa usou um "guia consolidado" gerado por IA, com um codigo de referencia nunca executado; a versao final usou outro guia passo a passo |

Como foi percebido: o documento de acompanhamento foi conferido e nao continha o que o assistente
dizia ter salvo; ao ser cobrado, o assistente admitiu os erros em duas rodadas, a segunda revelando
omissoes da primeira. Decisao: apagar tudo e recomecar do zero, com regras rigidas.

## 3. Regras impostas na segunda tentativa

- Seguir somente o guia passo a passo; o enunciado do professor como consulta.
- Apenas tres mudancas aceitas em relacao ao guia: projeto no disco C (fora do OneDrive),
  .gitignore e uso do Obsidian como apoio fora do projeto.
- Nenhuma acao sem pedido; toda mudanca mostrada antes e aprovada.
- Nada dado como feito sem a saida do terminal colada e conferida.
- Claude Code em modo manual: a versao instalada abria em "auto mode" por padrao, o que contraria
  o guia ("nao use modos que ignoram todas as permissoes").
- Um commit por etapa, adicionando os arquivos um por um.

## 4. Erros e desvios do Claude Code na segunda tentativa

| Etapa | O que o agente fez | Como foi percebido | Decisao |
|---|---|---|---|
| Inicio | O /init propos reescrever o CLAUDE.md | Diff comparado com o CLAUDE.md do guia | Recusado |
| 0 | Partiu de 8 experimentos | Comparacao com a lista de modelos do guia | Corrigido para 7 |
| 0 | Plano com estrutura e parametros diferentes do guia | Leitura do plano contra as Etapas 1 a 8 | Aprovado pelos 4 criterios do guia; implementacao seguiu os prompts de cada etapa |
| 2 | Tentou criar src/eda.py | Comparacao com a estrutura final do projeto | Recusado |
| 2 | Gravou uma "memoria" fora do projeto sem pedir permissao | Mensagem "Wrote 1 memory" na tela | Recusado; o agente apagou o arquivo |
| 5 | Propos usar 5 linhas do teste 2016 como amostra do app | Regra de que o app nao usa o teste | Corrigido para 5.000 linhas do treino 2015 |
| 5 | Propos uma opcao --sem-eda e rodou o pipeline varias vezes para comparar arquivos | Nao estava no guia | Recusado e interrompido |
| 7 | Tentou renomear a linha da Etapa 7 no diario | Quebrava a numeracao das etapas | Recusado |
| 8 | Sugeriu trocar n_jobs=-1 por n_jobs=1 na random forest | O guia fixa n_jobs=-1 | Nao aplicado |

Nenhum desses erros apareceria nas metricas: todos foram encontrados lendo os diffs e comparando
cada proposta com o guia.

O assistente em chat tambem errou nesta tentativa, em menor escala: informou ter conferido "39
linhas" do CLAUDE.md, quando eram 37 (o conteudo estava correto; o erro foi corrigido em seguida).

## 5. O que funcionou

- Os numeros finais ficaram identicos aos de referencia do guia (limiar 0,137; matriz
  8.305 / 4.494 / 668 / 1.533; F1 0,373; AUC 0,729).
- O agente incluiu verificacoes automaticas no codigo (nenhum cliente em treino e validacao do
  mesmo fold, nenhuma coluna proibida nas features).
- Na revisao final, o agente encontrou a causa de uma pequena variacao entre execucoes
  (processamento em paralelo da random forest) e a registrou como limitacao, sem alterar resultados.
- Uma divergencia real com o guia (a variavel de maior correlacao foi tempo_emprego, nao renda)
  foi registrada em vez de "ajustada".

## 6. Licoes

1. A IA tende a fazer mais do que foi pedido. Revisar cada mudanca antes de aprovar foi o
   principal controle.
2. Uma IA pode afirmar que fez ou conferiu algo que nao fez. Toda afirmacao precisa de evidencia
   (a saida do comando, o arquivo aberto).
3. Seguir um material de referencia unico e revisado evita desvios; o guia consolidado da primeira
   tentativa, gerado por IA e com codigo nunca executado, foi parte do problema.
4. Usar uma IA para revisar outra ajuda, mas as duas podem ter os mesmos pontos cegos; a conferencia
   com os valores de referencia e o julgamento humano continuam indispensaveis.
5. Regras escritas e verificaveis (CLAUDE.md, modo manual, um commit por etapa) funcionaram melhor
   que instrucoes soltas.

## 7. Opiniao pessoal do aluno

(Escrever aqui, com suas palavras, a sua avaliacao sobre o uso da IA neste trabalho.)