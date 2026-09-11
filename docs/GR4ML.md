# Caderno GR4ML — Precificação Assistida de Aluguel por Temporada em Copacabana

**Disciplina:** Engenharia de Software para IA e Frameworks Profundos — CIn/UFPE
**Fonte de dados:** Inside Airbnb — Rio de Janeiro, snapshot de 24/06/2026 (`listings.csv.gz`)

---

## 1. Contexto do projeto (Fase 0)

O **Gestor de Precificação** de uma administradora de imóveis de temporada em Copacabana precisa definir o preço da diária de cada imóvel do portfólio. Hoje essa decisão é tomada caso a caso, por comparação manual com anúncios vizinhos, sem critério objetivo e sem registro do raciocínio. O resultado são imóveis precificados acima do mercado — que ficam vagos — e imóveis abaixo do mercado — que ocupam mas deixam receita na mesa. Copacabana concentra 15.036 dos 48.713 anúncios ativos do Rio (30,9%), o que torna o bairro o caso de maior impacto e, ao mesmo tempo, o de maior dificuldade comparativa: há oferta demais para se comparar "no olho".

Essa concentração não é apenas expressiva no contexto da cidade: segundo levantamento da ONG internacional Inside Airbnb divulgado em julho de 2026, **Copacabana registra a maior densidade de anúncios de aluguel por temporada por quilômetro quadrado do mundo** — 2.852 anúncios por km² em um único trecho do bairro, muito à frente da segunda colocada, a Lapa, com 1.307 por km². No mesmo levantamento, a oferta de aluguel residencial tradicional no Rio aparece com queda de 31,8% entre 2023 e 2026, o que situa este projeto dentro de um debate urbano em curso sobre o efeito da locação de curta duração na moradia.

> **Nota sobre os números.** O levantamento divulgado à imprensa reporta 48.193 anúncios ativos no Rio e 13.174 em Copacabana. Minha extração direta do arquivo `listings.csv` encontrou 48.713 e 15.036, respectivamente — diferença atribuível ao recorte de coleta (o snapshot utilizado cobre 25/06 a 01/07/2026) e a possíveis filtros de anúncios inativos aplicados pela organização. A proporção, contudo, é praticamente idêntica: 30,94% na imprensa contra 30,9% na apuração deste trabalho.

O preço em Copacabana também não é estável ao longo do ano: réveillon, carnaval e feriados prolongados alteram substancialmente a disposição a pagar. O sistema trata essas duas dimensões **separadamente** — um componente de aprendizado de máquina estima o preço-base do imóvel a partir de suas características, e um componente determinístico sinaliza a pressão de demanda da data escolhida. A decisão de quanto ajustar permanece com o Gestor.

---

## 2. Business View (Fase 1)

### Passo 1.1 — Atores de negócio

| Nome do Ator | Papel na organização |
|---|---|
| Gestor de Precificação | Define e revisa o preço da diária dos imóveis do portfólio (**ator principal**) |
| Anfitrião Proprietário | Dono do imóvel; aprova o preço sugerido e cobra justificativa |
| Analista de Dados | Mantém a base de anúncios e opera o modelo |

### Passo 1.2 — Goal, Indicador e expectativas de qualidade

| Ator | Goal | Indicador | Expectativas de qualidade (informal) |
|---|---|---|---|
| Gestor de Precificação | Reduzir o número de imóveis precificados fora da faixa de mercado ao longo do próximo ano | % de anúncios do portfólio com preço divergindo mais de 25% da referência de mercado — atual: ~56% · meta: ≤ 40% | "Não posso sugerir um preço que faça o imóvel encalhar, nem um que o subvalorize." · "Preciso do valor da diária na hora em que cadastro o anúncio, não no dia seguinte." · "No réveillon o preço é outro. Não pode me dar o mesmo valor de uma terça-feira de agosto." |
| Anfitrião Proprietário | Entender por que seu imóvel vale aquele valor | Nº de contestações de preço por mês | "O proprietário sempre pergunta *por que* esse valor. Preciso conseguir responder." |
| Analista de Dados | Manter o processo auditável e barato | Tempo de retreino | "Se eu rodar duas vezes, tem que dar o mesmo resultado." |

> **Nota sobre o valor atual do indicador:** medido no próprio dataset. A regra de comparação manual mais próxima do que se faz hoje (mediana por capacidade) acerta dentro de ±25% em apenas 44,4% dos casos — logo, ~56% ficam fora da faixa.

### Passo 1.3 — Decision Goal

> O **Gestor de Precificação** precisa decidir **qual preço de diária publicar** para cada imóvel — entre *manter o preço atual*, *aumentar* ou *reduzir* — **a cada novo cadastro** e **na revisão mensal do portfólio**.

### Passo 1.4 — Question Goals

| Question Goal | Tipo | Tempo | Frequência |
|---|---|---|---|
| QG1 — Qual é o preço de diária esperado para um imóvel com estas características em Copacabana? | Quanto | Presente | Sob demanda (a cada novo cadastro) |
| QG2 — Quais atributos do imóvel mais influenciam o preço praticado no bairro? | O quê | Presente | Trimestral |
| QG3 — Este anúncio está precificado acima ou abaixo da referência de mercado? | Se/quanto | Presente | Mensal, para todo o portfólio |
| QG4 — Em que datas a pressão de demanda justifica ajuste no preço? | Quando | Futuro (próximos 12 meses) | Mensal, ao publicar o calendário |

---

## 3. Analytics Design View (Fase 2)

### Passo 2.1 — Tipo de análise

| Question Goal | Tipo de análise | Justificativa |
|---|---|---|
| QG1 | Preditiva | Estima um valor numérico desconhecido (o preço justo) a partir de atributos observáveis |
| QG2 | Descritiva | Resume e explica a relação entre atributos e preço no conjunto existente |
| QG3 | Preditiva | Compara o preço praticado com a estimativa do modelo; o desvio é o resultado |
| QG4 | Descritiva | Resume o comportamento já configurado pelos anfitriões no calendário; não estima valor desconhecido |

### Passo 2.2 — Tarefa de ML e algoritmos candidatos

| Meta Analítica | Tarefa de ML | Algoritmos candidatos |
|---|---|---|
| Estimar o preço da diária (QG1, QG3) | Regressão supervisionada | 1. Mediana global (regra) · 2. Mediana por capacidade e tipo (regra) · 3. Regressão Linear · 4. **Rede Neural MLP (PyTorch)** |
| Identificar atributos determinantes (QG2) | Análise de importância de atributos | Correlação de Pearson · coeficientes da regressão linear |
| Quantificar a pressão de demanda por data (QG4) | **Nenhuma — regra determinística** | Índice calculado sobre `minimum_nights` mediano por data e ocupação relativa a janela móvel de ±21 dias |

### Passo 2.3 — Softgoals formalizados

| Softgoal | Origem |
|---|---|
| SG1 — Confiabilidade da estimativa | Stakeholder: "não posso sugerir um preço que faça o imóvel encalhar, nem um que o subvalorize" |
| SG2 — Explicabilidade da estimativa | Stakeholder: "o proprietário sempre pergunta por que esse valor" |
| SG3 — Resposta imediata | Stakeholder: "preciso do valor na hora em que cadastro o anúncio" |
| SG4 — Reprodutibilidade | Técnico (Analista de Dados): "se eu rodar duas vezes, tem que dar o mesmo resultado" |
| SG5 — Baixo custo de treino e operação | Técnico: a administradora não tem infraestrutura de GPU |

### Passo 2.4 — Softgoal × Métrica, por algoritmo

| Algoritmo | SG1 MAE (R$) | SG2 Explicab. | SG3 Inferência | SG4 Reprod. | SG5 Custo CPU |
|---|---|---|---|---|---|
| Mediana global | 300,37 | Total | Imediato | Garantida | Nenhum |
| Mediana cap.+tipo | 259,29 | Total | Imediato | Garantida | Nenhum |
| Regressão linear | a medir | Alta | < 1 s | Garantida | Segundos |
| **Rede neural MLP** | a medir | Média | < 1 s | Semente fixa | Minutos |

**Conclusão — qual algoritmo escolhi e por quê:**

Escolhi a **Rede Neural MLP em PyTorch**, mantendo os três outros como baselines de comparação obrigatória. A justificativa não é "rede neural é melhor": é que a relação entre capacidade e preço observada em Copacabana **não é linear** — o preço mediano sai de R$ 229 (1 hóspede) para R$ 438 (4 hóspedes) e R$ 1.065 (8 hóspedes), com inclinação crescente. Uma regressão linear subestima os imóveis grandes e superestima os pequenos. A MLP captura essa curvatura combinando múltiplos atributos simultaneamente, o que a tabela de consulta por capacidade não faz.

Assumo explicitamente o custo dessa escolha: perde-se explicabilidade direta em relação à regressão linear (SG2). Mitigo isso reportando a importância dos atributos calculada à parte (QG2).

### Passo 2.5 — Conexão de volta ao indicador de negócio

Se o modelo reduzir o MAE dos atuais R$ 259 (melhor regra sem ML) para a faixa de R$ 230, e elevar a proporção de estimativas dentro de ±25% do preço real dos atuais 44,4% para acima de 60%, o percentual de anúncios precificados fora da faixa de mercado cai de ~56% para ~40% — atingindo a meta do Passo 1.2.

---

## 4. Data Preparation View (Fase 3)

### Passo 3.1 — Fontes de dados

| Fonte de dados | Principais campos/atributos |
|---|---|
| `listings.csv` — Imóvel (Inside Airbnb) | `id`, `neighbourhood_cleansed`, `price`, `accommodates`, `bedrooms`, `beds`, `bathrooms_text`, `room_type`, `property_type`, `latitude`, `longitude`, `amenities` |
| `listings.csv` — Anfitrião | `host_id`, `host_is_superhost`, `calculated_host_listings_count` |
| `listings.csv` — Avaliações (agregadas) | `number_of_reviews`, `review_scores_rating`, `review_scores_location`, `reviews_per_month` |
| `listings.csv` — Disponibilidade | `minimum_nights`, `availability_365`, `estimated_occupancy_l365d` |
| `calendar.csv` — Calendário diário | `listing_id`, `date`, `available`, `minimum_nights`, `maximum_nights` — 17,8 milhões de linhas, cobrindo 25/06/2026 a 30/06/2027 |

### Passo 3.2 — Data Operations (transformações, na ordem)

| Nº | Transformação aplicada | Origem / efeito |
|---|---|---|
| 1 | Filtrar `neighbourhood_cleansed == 'Copacabana'` | 48.713 → 15.036 registros |
| 2 | Converter `price` de texto (`"$354.00"`) para float, removendo `$` e separador de milhar | Coluna alvo utilizável |
| 3 | Descartar registros sem preço | 15.036 → 13.768 (−8,4%) |
| 4 | Remover outliers fora da faixa R$ 100–5.000 | 13.768 → 13.558 (preserva 98,5%); elimina o máximo espúrio de R$ 574.013 |
| 5 | Extrair o número de banheiros de `bathrooms_text` | `bathrooms` tem 14,3% de nulos; `bathrooms_text` tem 0,1% |
| 6 | Imputar nulos de `bedrooms` e `beds` pela mediana | Cobre 13,2% e 10,2% de ausências |
| 7 | Descartar colunas 100% vazias (`instant_bookable`), identificadores e URLs | Remove ruído e vazamento de identificador |
| 8 | One-Hot Encoding de `room_type` e de `property_type` agrupado | 4 e ~6 categorias, respectivamente |
| 9 | Padronização (z-score) das variáveis numéricas | Escalas incompatíveis: latitude ~−22 vs. `availability_365` ~365 |
| 10 | Divisão treino/teste 80/20 com semente fixa | Reprodutibilidade (SG4) |
| 11 | **(calendário)** Filtrar o `calendar.csv` pelos `listing_id` de Copacabana | 17,8 mi → subconjunto do bairro |
| 12 | **(calendário)** Agregar por data: mediana de `minimum_nights` e proporção de imóveis com `available = 'f'` | 371 datas |
| 13 | **(calendário)** Normalizar a ocupação por janela móvel de ±21 dias | Neutraliza o viés de horizonte de calendário (ver Reflexão crítica) |
| 14 | **(calendário)** Calcular o Índice de Pressão de Demanda e classificar cada data em normal · elevada · alta · crítica | Tabela de 371 datas, artefato do componente não-ML |

### Passo 3.3 — Dataset final

| Coluna (feature) | Descrição | De qual fonte / transformação |
|---|---|---|
| `accommodates` | Capacidade de hóspedes | Imóvel · padronizada (T9) |
| `bedrooms` | Nº de quartos | Imóvel · imputada (T6) + padronizada |
| `beds` | Nº de camas | Imóvel · imputada (T6) + padronizada |
| `bathrooms` | Nº de banheiros | Derivada de `bathrooms_text` (T5) + padronizada |
| `room_type_*` | Tipo de acomodação | One-Hot (T8) |
| `property_type_*` | Tipo de imóvel agrupado | One-Hot (T8) |
| `minimum_nights` | Mínimo de noites | Disponibilidade · padronizada |
| `availability_365` | Dias disponíveis no ano | Disponibilidade · padronizada |
| `number_of_reviews` | Nº de avaliações | Avaliações · padronizada |
| `review_scores_rating` | Nota geral | Avaliações · imputada + padronizada |
| `host_is_superhost` | Anfitrião é superhost | Anfitrião · binária |
| **`price`** | **Preço da diária em R$ — variável alvo** | Imóvel · convertida (T2) e filtrada (T3, T4) |

**Unidade de análise:** um imóvel anunciado em Copacabana. **13.558 linhas.**

**Segundo artefato — tabela de pressão de demanda.** Unidade de análise: uma data. **371 linhas**, cobrindo 25/06/2026 a 30/06/2027, com as colunas `date`, `min_noites_mediano`, `taxa_ocupacao`, `ipd` e `faixa`. Essa tabela alimenta o componente não-ML e é independente do modelo — não passa por treino.

---

## 5. Frase de verificação final

> Para ajudar o **Gestor de Precificação** a decidir **qual preço de diária publicar para cada imóvel em Copacabana**, respondendo às perguntas **"qual é o preço esperado para um imóvel com estas características?"** e **"em que datas a pressão de demanda justifica ajuste?"**, vou usar uma **Rede Neural MLP implementada em PyTorch** para o preço-base, combinada com um **componente determinístico de calendário** para a pressão de demanda, satisfazendo os softgoals **Confiabilidade, Resposta imediata, Reprodutibilidade e Baixo custo**, medidos por **MAE em reais, percentual de acertos dentro de ±25%, tempo de inferência e tempo de treino em CPU**, usando o **dataset preparado de 13.558 imóveis com 11 atributos** e a **tabela de 371 datas**, ambos vindos do **snapshot do Inside Airbnb de junho de 2026**. Isso deve impactar o indicador **percentual de anúncios precificados fora da faixa de mercado**, reduzindo-o de **~56% para ~40%**.

---

## 6. Requisitos consolidados (Fase 4)

### 6.1 Requisitos Funcionais

| ID | Requisito Funcional | Origem no modelo |
|---|---|---|
| RF01 | O sistema deve carregar a base de anúncios do Inside Airbnb e selecionar os imóveis do bairro Copacabana, para permitir que o Gestor de Precificação trabalhe apenas com o mercado relevante. | Decision Goal |
| RF02 | O sistema deve realizar o pré-processamento dos dados, convertendo o preço textual em numérico, tratando valores ausentes e removendo outliers fora da faixa R$ 100–5.000. | QG1 · Fase 3 |
| RF03 | O sistema deve dividir os dados em conjuntos de treino e teste na proporção 80/20, com semente fixa. | SG4 · Fase 3 |
| RF04 | O sistema deve treinar uma rede neural em PyTorch para estimar o preço da diária a partir dos atributos do imóvel. | QG1 |
| RF05 | O sistema deve avaliar o modelo no conjunto de teste, reportando MAE em reais e o percentual de estimativas dentro de ±25% do preço real. | QG1 · SG1 |
| RF06 | O sistema deve comparar o desempenho do modelo com pelo menos dois baselines sem aprendizado de máquina (mediana global e mediana por capacidade e tipo). | Passo 2.4 |
| RF07 | O sistema deve estimar o preço de diária para um imóvel novo, informado pelo usuário, para permitir que o Gestor decida o valor a publicar. | Decision Goal · QG1 |
| RF08 | O sistema deve salvar o modelo treinado em disco e carregá-lo posteriormente sem necessidade de novo treinamento. | SG3 · SG5 |
| RF09 | O sistema deve calcular, a partir do calendário diário, um Índice de Pressão de Demanda para cada data dos próximos 12 meses, classificando-a em normal, elevada, alta ou crítica. | QG4 |
| RF10 | O sistema deve apresentar ao usuário o preço-base estimado acompanhado da faixa de pressão de demanda da data escolhida, mantendo com ele a decisão sobre o ajuste final. | Decision Goal · QG4 |

### 6.2 Requisitos Não-Funcionais

| ID | Requisito Não-Funcional | Origem | Critério de aceitação |
|---|---|---|---|
| RNF01 | O sistema deve garantir confiabilidade da estimativa, medida pelo MAE no conjunto de teste, inferior ao melhor baseline sem ML (R$ 259,29). | SG1 | Execução da avaliação final no conjunto de teste |
| RNF02 | O sistema deve garantir que ao menos 60% das estimativas fiquem dentro de ±25% do preço real. | SG1 | Cálculo do percentual no conjunto de teste |
| RNF03 | O sistema deve garantir reprodutibilidade: duas execuções com a mesma semente devem produzir métricas idênticas. | SG4 | Executar o pipeline duas vezes e comparar as métricas |
| RNF04 | O sistema deve responder a uma estimativa individual em menos de 1 segundo. | SG3 | Medição do tempo de inferência |
| RNF05 | O sistema deve treinar completamente em CPU em menos de 5 minutos. | SG5 | Medição do tempo de treino |
| RNF06 | O sistema deve ser modular, com carregamento, pré-processamento, modelo, treino, avaliação e inferência em módulos independentes e testáveis. | Técnico | Estrutura do repositório e execução da suíte de testes |
| RNF07 | O componente de calendário deve ser determinístico e independente do modelo: alterações no treinamento não podem alterar o Índice de Pressão de Demanda. | Técnico · SG2 · SG4 | Executar o cálculo do índice sem o modelo carregado e comparar os resultados |

### 6.3 Requisitos de Dados

| ID | Requisito de Dados | RF que alimenta |
|---|---|---|
| RD01 | O sistema deve coletar os dados de anúncios do snapshot do Inside Airbnb para o Rio de Janeiro, filtrando o bairro Copacabana. | RF01 |
| RD02 | O sistema deve tratar a coluna `price`, convertendo-a de texto para valor numérico e descartando os 8,4% de registros sem preço informado. | RF02 |
| RD03 | O sistema deve tratar os valores ausentes de `bedrooms` (13,2%), `beds` (10,2%) e `bathrooms` (14,3%), derivando o número de banheiros de `bathrooms_text` e imputando os demais pela mediana. | RF02 |
| RD04 | O sistema deve aplicar One-Hot Encoding às variáveis categóricas e padronização às numéricas antes do treinamento. | RF02 · RF04 |
| RD05 | O sistema deve coletar o calendário diário dos imóveis de Copacabana e tratar o campo `available`, reconhecendo que ele mistura *reservado* com *calendário não aberto pelo anfitrião*, corrigindo o viés por janela móvel. | RF09 |

### 6.4 Critério de Aceitação do conjunto

> Este conjunto de requisitos será considerado bem-sucedido se o percentual de anúncios do portfólio precificados fora da faixa de ±25% da referência de mercado cair de ~56% para no máximo 40%, observado na revisão mensal do portfólio ao longo de um ano.

---

## 7. Reflexão crítica

**O ML era realmente necessário aqui?** Parcialmente — e a formalização foi justamente o que permitiu enxergar isso. Antes de escrever qualquer modelo, medi três alternativas sem aprendizado de máquina no próprio dataset: prever sempre a mediana global erra R$ 300,37 em média; uma tabela de consulta por capacidade erra R$ 262,15; acrescentando o tipo de acomodação, R$ 259,29. Ou seja, **a maior parte do ganho possível vem de uma regra simples**, que qualquer planilha implementaria. O espaço que sobra para o ML é a diferença entre R$ 259 e o erro irredutível do problema — e essa margem é modesta.

Isso não invalida o projeto, mas redefine o critério de sucesso: o modelo não precisa apenas "funcionar", precisa **superar a tabela de consulta**. Sem essa medição, eu teria declarado sucesso ao obter um MAE de R$ 280 — que é, na verdade, pior que não usar ML nenhum. Esse é o risco que a Lei de Goodhart descreve: otimizar a métrica sem verificar se ela representa o objetivo.

**O que a formalização dos Softgoals revelou que não estava claro na conversa inicial?** Duas coisas. Primeiro, o conflito entre **Confiabilidade e Explicabilidade**: o stakeholder pediu as duas sem perceber que são concorrentes na escolha do algoritmo — a regressão linear entrega explicação direta, a rede neural entrega (potencialmente) menos erro. A conversa inicial tratava as duas como se fossem gratuitas. Segundo, a expectativa "preciso do valor na hora" parecia um detalhe de conveniência, mas ao virar métrica (tempo de inferência < 1 s) revelou um requisito de arquitetura: o modelo precisa ser **carregado de disco já treinado**, e não retreinado a cada consulta — o que originou o RF08.

Uma terceira revelação veio dos dados, não do stakeholder: a nota das avaliações tem correlação praticamente nula com o preço (0,006). O Gestor supunha que imóveis mais bem avaliados cobrassem mais. Não é o que os dados de Copacabana mostram — o que sugere que o preço é definido pelo **tamanho do imóvel**, e a reputação afeta a ocupação, não a diária.

**O que a exigência de sazonalidade revelou sobre os dados.** A pergunta "como o sistema lida com o réveillon" expôs que o `listings.csv` é uma **fotografia de um único instante** — coletado entre 25/06 e 01/07/2026, com um único preço por imóvel. Pior: é uma foto de baixa temporada, em pleno inverno e sem eventos. Um modelo treinado só nesses dados aprende o piso do mercado e subprecificaria sistematicamente em dezembro. Foi essa constatação que originou a separação entre preço-base e pressão de demanda.

Ao buscar a sazonalidade no `calendar.csv`, encontrei duas coisas. Primeiro, que **o arquivo não contém preço** — as versões recentes do Inside Airbnb removeram as colunas `price` e `adjusted_price` do calendário. Isso inviabilizou a ideia inicial de calcular um multiplicador de preço por data e me obrigou a redefinir o componente como um índice de *pressão*, não de preço. Preferi entregar um sinal honesto e deixar o ajuste com o humano a converter pressão em preço por meio de uma constante arbitrada por mim.

Segundo, um **viés de horizonte** que quase passou despercebido: a taxa de indisponibilidade sobe de 41,7% nos próximos 30 dias para 68,1% no horizonte de 271 a 371 dias. Como é implausível que datas mais distantes estejam mais reservadas, concluí que o campo `available` mistura *reservado* com *calendário ainda não aberto*. Usar ocupação bruta como medida de demanda produziria um índice que cresce com a distância no tempo — um artefato puro. A correção por janela móvel de ±21 dias compara cada data apenas com suas vizinhas, neutralizando o efeito. Já `minimum_nights` mostrou-se estável ao longo do horizonte, por ser decisão do anfitrião e não consequência de reservas, e por isso é o sinal principal do índice.

**Limitação declarada.** O índice só reflete o que os anfitriões **já configuraram**. Réveillon (3,00 noites mínimas contra base de 2,22) e Carnaval (2,77) aparecem com clareza, assim como feriados prolongados como 7 de setembro (2,78) e 12 de outubro (2,64). Mas o feriado de 1º de maio de 2027 apareceu em 2,26 — praticamente indistinguível de uma data comum. A explicação mais provável é que shows e eventos de grande porte são anunciados com poucos meses de antecedência, e o calendário foi coletado em junho de 2026. O sistema, portanto, **não antecipa eventos ainda não anunciados** — o que reforça a decisão de manter o humano no circuito em vez de automatizar o ajuste de preço.
