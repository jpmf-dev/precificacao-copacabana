"""Limpeza dos dados brutos (transformações T2 a T7).

Cada função resolve um problema de qualidade identificado na Data
Preparation View e pode ser testada isoladamente.
"""

import re

import numpy as np
import pandas as pd

from src import config


def converter_preco(serie: pd.Series) -> pd.Series:
    """Converte o preço de texto para número (transformação T2).

    A base traz o preço no formato `"$1,234.00"`. É preciso remover o
    cifrão e o separador de milhar antes de converter para float.

    Args:
        serie: coluna de preços em formato textual.

    Returns:
        Série de floats, com NaN onde a conversão não foi possível.
    """
    limpa = (
        serie.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(limpa, errors="coerce")


def extrair_banheiros(serie: pd.Series) -> pd.Series:
    """Extrai o número de banheiros do campo textual (transformação T5).

    A coluna `bathrooms` tem 14,3% de valores ausentes, enquanto
    `bathrooms_text` tem apenas 0,1%. O texto vem em formatos como
    "1 bath", "2.5 baths" ou "Half-bath" — este último equivale a 0,5.

    Args:
        serie: coluna `bathrooms_text`.

    Returns:
        Série de floats com a quantidade de banheiros.
    """

    def _extrair(valor: object) -> float:
        if not isinstance(valor, str):
            return np.nan
        if "half" in valor.lower():
            return 0.5
        achado = re.search(r"(\d+\.?\d*)", valor)
        return float(achado.group(1)) if achado else np.nan

    return serie.map(_extrair).astype(float)


def filtrar_faixa_preco(
    df: pd.DataFrame,
    minimo: float = config.PRECO_MINIMO,
    maximo: float = config.PRECO_MAXIMO,
) -> pd.DataFrame:
    """Remove registros sem preço e outliers (transformações T3 e T4).

    A base contém anúncios com preço de até R$ 574.013, valores que
    correspondem a anúncios inativos ou erros de cadastro. A faixa padrão
    preserva 98,5% dos registros com preço informado.

    Args:
        df: DataFrame com a coluna de preço já convertida para número.
        minimo: preço mínimo aceito, inclusive.
        maximo: preço máximo aceito, inclusive.

    Returns:
        Cópia do DataFrame apenas com os registros dentro da faixa.

    Raises:
        ValueError: se `minimo` não for menor que `maximo`.
    """
    if minimo >= maximo:
        raise ValueError(f"Faixa inválida: mínimo ({minimo}) >= máximo ({maximo}).")

    alvo = config.COLUNA_ALVO
    com_preco = df.dropna(subset=[alvo])
    dentro = com_preco[(com_preco[alvo] >= minimo) & (com_preco[alvo] <= maximo)]
    return dentro.copy()


def imputar_por_mediana(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Preenche valores ausentes com a mediana da coluna (transformação T6).

    A mediana é preferida à média por ser robusta a outliers, que são
    frequentes em atributos como número de quartos e camas.

    Args:
        df: DataFrame de origem.
        colunas: nomes das colunas a imputar. Colunas ausentes são ignoradas.

    Returns:
        Cópia do DataFrame com os ausentes preenchidos.
    """
    resultado = df.copy()
    for coluna in colunas:
        if coluna not in resultado.columns:
            continue
        mediana = resultado[coluna].median()
        if pd.isna(mediana):
            mediana = 0.0
        resultado[coluna] = resultado[coluna].fillna(mediana)
    return resultado


def preparar_anuncios(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a sequência completa de limpeza dos anúncios.

    Encadeia as transformações T2 a T6 na ordem definida na Data
    Preparation View.

    Args:
        df: DataFrame de anúncios já filtrado por bairro.

    Returns:
        DataFrame limpo, pronto para a construção das features.
    """
    limpo = df.copy()
    limpo[config.COLUNA_ALVO] = converter_preco(limpo[config.COLUNA_ALVO])

    if "bathrooms_text" in limpo.columns:
        limpo["bathrooms"] = extrair_banheiros(limpo["bathrooms_text"])

    limpo = filtrar_faixa_preco(limpo)
    limpo = imputar_por_mediana(limpo, config.COLUNAS_NUMERICAS)
    return limpo
