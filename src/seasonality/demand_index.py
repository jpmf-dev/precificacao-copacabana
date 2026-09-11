"""Índice de Pressão de Demanda — componente determinístico (RF09).

Este módulo não contém aprendizado de máquina. É uma regra de negócio
calculada sobre o calendário, mantida separada do modelo para que
retreinar a rede não altere o índice (RNF07).

Duas decisões de projeto merecem destaque:

1. O sinal principal é `minimum_nights`, não a taxa de ocupação. O mínimo
   de noites é uma decisão do anfitrião e se mostrou estável ao longo do
   horizonte; a ocupação, não.

2. A ocupação entra apenas como ajuste, normalizada por uma janela móvel.
   O campo `available` mistura "reservado" com "calendário ainda não
   aberto pelo anfitrião", o que faz a indisponibilidade crescer com a
   distância no tempo — de 41,7% nos próximos 30 dias para 68,1% no
   horizonte de 271 a 371 dias. Comparar cada data apenas com as vizinhas
   remove esse artefato.
"""

import numpy as np
import pandas as pd

from src import config

# Limiares do índice. São uma decisão de projeto, não um resultado dos dados:
# foram calibrados sobre a distribuição observada em Copacabana (IPD entre
# 0,90 e 1,26) de modo que o réveillon e os fins de semana do Rock in Rio
# caiam em "crítica" e o carnaval em "alta". Alterá-los muda a classificação,
# não o índice — por isso ficam isolados aqui.
FAIXAS: dict[str, float] = {
    "normal": 1.05,
    "elevada": 1.12,
    "alta": 1.20,
}


def agregar_por_data(df_calendario: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    """Resume o calendário diário por data (transformação T12).

    O mínimo de noites é agregado pela **média truncada**, não pela mediana.
    A mediana assume apenas valores inteiros baixos (2 ou 3 noites), o que
    tornava o índice praticamente binário: ou a data era comum, ou saltava
    direto para crítica, sem faixas intermediárias. A média capta variações
    graduais — uma data em que 30% dos anfitriões subiram a exigência fica
    entre as duas situações, que é o comportamento desejado.

    O truncamento em `config.MAX_NOITES` é necessário porque a base contém
    anúncios de aluguel mensal disfarçado, com exigências de até 666 noites.
    São apenas 0,41% dos registros, mas sem o corte distorceriam a média:
    ela cai de 3,38 para 3,02 quando a cauda é limitada.

    Args:
        df_calendario: calendário bruto com listing_id, date, available
            e minimum_nights.
        ids: identificadores dos imóveis do bairro de interesse.

    Returns:
        DataFrame com uma linha por data, contendo `min_noites_medio`,
        `taxa_ocupacao` e `anuncios`.

    Raises:
        ValueError: se nenhum registro restar após o filtro por bairro.
    """
    recorte = df_calendario[df_calendario["listing_id"].isin(ids)].copy()
    if recorte.empty:
        raise ValueError("Nenhum registro de calendário para os imóveis informados.")

    recorte["ocupado"] = (recorte["available"] == "f").astype(int)
    recorte["noites"] = recorte["minimum_nights"].clip(upper=config.MAX_NOITES)

    agregado = (
        recorte.groupby("date")
        .agg(
            min_noites_medio=("noites", "mean"),
            taxa_ocupacao=("ocupado", "mean"),
            anuncios=("ocupado", "size"),
        )
        .reset_index()
    )
    agregado["date"] = pd.to_datetime(agregado["date"])
    return agregado.sort_values("date").reset_index(drop=True)


def normalizar_por_janela(
    serie: pd.Series,
    janela_dias: int = config.JANELA_SAZONAL_DIAS,
) -> pd.Series:
    """Divide cada valor pela mediana de sua vizinhança temporal (T13).

    Args:
        serie: série ordenada por data.
        janela_dias: raio da janela, em dias, para cada lado.

    Returns:
        Série de razões, onde 1,0 significa "igual às datas vizinhas".

    Raises:
        ValueError: se a janela não for positiva.
    """
    if janela_dias <= 0:
        raise ValueError(f"Janela deve ser positiva, recebido {janela_dias}.")

    largura = 2 * janela_dias + 1
    referencia = serie.rolling(window=largura, center=True, min_periods=1).median()
    referencia = referencia.replace(0.0, np.nan)
    return (serie / referencia).fillna(1.0)


def classificar(indice: float) -> str:
    """Converte o índice numérico em faixa legível.

    Args:
        indice: valor do Índice de Pressão de Demanda.

    Returns:
        Uma entre "normal", "elevada", "alta" ou "crítica".
    """
    for faixa, limite in FAIXAS.items():
        if indice < limite:
            return faixa
    return "crítica"


def construir_indice(
    df_calendario: pd.DataFrame,
    ids: set[int],
    janela_dias: int = config.JANELA_SAZONAL_DIAS,
    cobertura_minima: float = config.COBERTURA_MINIMA,
) -> pd.DataFrame:
    """Constrói a tabela de pressão de demanda por data (T12 a T14).

    O índice combina dois sinais: a razão do mínimo de noites médio em relação à
    mediana anual (peso dominante) e a ocupação normalizada pela janela
    móvel (ajuste).

    Datas com cobertura amostral insuficiente são descartadas. Cada anúncio
    publica 365 dias a partir da data em que foi coletado, e a coleta se
    espalha por vários dias — então as datas extremas do horizonte são
    observadas por uma fração pequena e enviesada dos imóveis (cerca de
    1.100 contra 13.900 no miolo). Calcular o índice sobre essa minoria
    produzia falsos picos nos últimos dias do calendário.

    Args:
        df_calendario: calendário bruto.
        ids: identificadores dos imóveis do bairro.
        janela_dias: raio da janela de normalização.
        cobertura_minima: fração do número máximo de anúncios por data
            que uma data precisa atingir para ser mantida.

    Returns:
        DataFrame com date, min_noites_medio, taxa_ocupacao, anuncios,
        ipd e faixa, restrito às datas com cobertura suficiente.

    Raises:
        ValueError: se a cobertura mínima não estiver em (0, 1].
    """
    if not 0.0 < cobertura_minima <= 1.0:
        raise ValueError(
            f"Cobertura mínima deve estar em (0, 1], recebido {cobertura_minima}."
        )

    agregado = agregar_por_data(df_calendario, ids)

    limite = cobertura_minima * agregado["anuncios"].max()
    agregado = agregado[agregado["anuncios"] >= limite].reset_index(drop=True)

    base_noites = float(agregado["min_noites_medio"].median())
    base_noites = base_noites if base_noites > 0 else 1.0
    razao_noites = agregado["min_noites_medio"] / base_noites

    razao_ocupacao = normalizar_por_janela(agregado["taxa_ocupacao"], janela_dias)

    agregado["ipd"] = (0.7 * razao_noites + 0.3 * razao_ocupacao).round(3)
    agregado["faixa"] = agregado["ipd"].map(classificar)
    return agregado


def consultar_data(df_indice: pd.DataFrame, data: str) -> dict[str, object]:
    """Recupera o índice de uma data específica (apoio ao RF10).

    Args:
        df_indice: tabela produzida por `construir_indice`.
        data: data no formato ISO, por exemplo "2026-12-31".

    Returns:
        Dicionário com a data, o índice e a faixa.

    Raises:
        KeyError: se a data não estiver coberta pelo calendário.
    """
    alvo = pd.to_datetime(data)
    linha = df_indice[df_indice["date"] == alvo]
    if linha.empty:
        inicio = df_indice["date"].min().date()
        fim = df_indice["date"].max().date()
        raise KeyError(f"Data {data} fora do calendário disponível ({inicio} a {fim}).")

    registro = linha.iloc[0]
    return {
        "data": str(registro["date"].date()),
        "ipd": float(registro["ipd"]),
        "faixa": str(registro["faixa"]),
    }
