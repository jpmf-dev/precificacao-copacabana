# Precificação Assistida — Aluguel por Temporada em Copacabana

Estima o preço-base da diária de imóveis de temporada em Copacabana e sinaliza
a pressão de demanda da data escolhida, mantendo a decisão final com o gestor.

Projeto da disciplina **Engenharia de Software para IA e Frameworks Profundos** —
CIn/UFPE.

## O problema

Proprietários e administradoras de imóveis em Copacabana definem o preço da
diária por comparação manual com anúncios vizinhos, sem critério objetivo.
Erram para cima e o imóvel encalha; erram para baixo e perdem receita.

Copacabana concentra 15.036 dos 48.713 anúncios ativos do Rio (30,9%) e tem a
maior densidade de anúncios de temporada por quilômetro quadrado do mundo —
o que torna a comparação manual inviável na prática.

## Arquitetura

O sistema tem dois componentes independentes:

| Componente | Tipo | Responsabilidade |
|---|---|---|
| Rede neural MLP | Aprendizado de máquina | Estima o preço-base a partir dos atributos do imóvel |
| Índice de Pressão de Demanda | Regra determinística | Classifica a data em normal, elevada, alta ou crítica |

Eles nunca se comunicam durante o processamento: retreinar o modelo não altera
o índice. A combinação acontece apenas na apresentação ao usuário.

## Estrutura

```
.
├── data/
│   ├── raw/                  dados brutos (não versionados)
│   └── processed/            índice de demanda gerado
├── docs/                     caderno GR4ML e diagramas
├── models/                   modelo treinado (.pth, não versionado)
├── src/
│   ├── config.py             parâmetros centrais
│   ├── data/loader.py        leitura dos arquivos
│   ├── preprocessing/
│   │   ├── cleaning.py       limpeza (T2 a T6)
│   │   └── features.py       matriz, split e padronização (T8 a T10)
│   ├── models/network.py     arquitetura da rede
│   ├── training/trainer.py   laço de treino
│   ├── evaluation/
│   │   ├── metrics.py        MAE e proporção dentro de ±25%
│   │   └── baselines.py      referências sem ML
│   ├── seasonality/
│   │   └── demand_index.py   índice de pressão de demanda
│   └── inference/predictor.py  persistência e estimativa
├── tests/                    suíte unittest
├── main.py                   orquestração
└── requirements.txt
```

A separação segue a responsabilidade de cada etapa do pipeline. `loader` só lê,
`cleaning` só limpa, `network` só descreve a arquitetura. Nenhum módulo conhece
o próximo, o que permite testar cada um isoladamente e trocar a implementação de
um sem tocar nos demais.

## Instalação

```bash
git clone <url-do-repositorio>
cd <pasta>
pip install -r requirements.txt
```

Baixe os dados em https://insideairbnb.com/get-the-data, seção **Rio de
Janeiro**, e coloque em `data/raw/`:

- `listings.csv.gz` (Detailed Listings data)
- `calendar.csv.gz` (Detailed Calendar Data)

Os arquivos não são versionados por causa do tamanho (23 MB e 43 MB comprimidos).

## Uso

```bash
python main.py treinar    # pipeline completo: dados, baselines, treino, avaliação
python main.py sazonal    # constrói o índice de pressão de demanda
python main.py estimar    # exemplo de estimativa para um imóvel novo
```

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes que dependem do PyTorch são ignorados automaticamente em ambientes sem
a biblioteca instalada, sem quebrar a suíte.

## Decisões de projeto

**Por que MAE e não RMSE.** O MAE é simétrico e está na mesma unidade do
problema — reais por diária. O gestor considera custosos tanto superestimar
quanto subestimar, então uma métrica que pune igualmente os dois lados
representa melhor o objetivo.

**Por que baselines sem ML são obrigatórios.** Prever sempre a mediana erra
cerca de R$ 300; uma tabela de consulta por capacidade e tipo erra cerca de
R$ 260. Sem essas referências, um modelo com MAE de R$ 280 pareceria bom sendo
pior que não usar ML. O modelo precisa superar a tabela, não apenas "funcionar".

**Por que salvar apenas o `state_dict`.** Serializar o modelo inteiro o acopla à
estrutura de diretórios e às classes do projeto, quebrando após refatorações.
Média e desvio da padronização são salvos junto: sem eles, um imóvel novo seria
normalizado com estatísticas diferentes das do treino.

**Por que o índice de demanda não é um multiplicador de preço.** O arquivo de
calendário do Inside Airbnb não contém preço nas versões recentes. Converter
pressão em preço exigiria uma constante arbitrada, e apresentar um chute como
resultado seria desonesto. O sistema entrega o sinal e mantém o humano no
circuito.

## Limitações conhecidas

- Os anúncios vêm de um único instante (junho de 2026), em baixa temporada. O
  preço-base reflete o piso do mercado.
- O índice de demanda só captura o que os anfitriões já configuraram. Réveillon
  e Carnaval aparecem; eventos anunciados com pouca antecedência, não.
- O campo `available` do calendário mistura "reservado" com "calendário não
  aberto". A normalização por janela móvel mitiga, mas não elimina, o efeito.
- Datas nas bordas do horizonte são observadas por poucos imóveis e são
  descartadas por cobertura insuficiente.
- Os limiares que separam as faixas do índice são uma decisão de projeto,
  calibrada sobre a distribuição observada em Copacabana. Não decorrem dos
  dados e precisariam ser revistos para outro bairro ou outro período.

## Dados

Inside Airbnb, snapshot do Rio de Janeiro de junho de 2026. Licença CC BY 4.0.
