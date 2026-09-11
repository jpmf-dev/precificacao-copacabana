"""Baselines sem aprendizado de máquina.

Existem para responder a uma pergunta que o projeto se recusa a ignorar:
o modelo de fato supera uma regra simples? Sem esses números, um MAE de
R$ 280 pareceria bom — quando na verdade é pior que uma tabela de consulta.
"""

import numpy as np
import pandas as pd

from src import config


def baseline_mediana_global(
    treino: np.ndarray,
    teste: np.ndarray,
) -> np.ndarray:
    """Prevê sempre a mediana do conjunto de treino.

    Args:
        treino: valores de preço do conjunto de treino.
        teste: valores de preço do conjunto de teste, usado só para o shape.

    Returns:
        Vetor constante com a mediana do treino, no tamanho do teste.
    """
    mediana = float(np.median(treino))
    return np.full(shape=len(teste), fill_value=mediana, dtype=float)


def baseline_mediana_por_grupo(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    colunas_grupo: list[str],
) -> np.ndarray:
    """Prevê a mediana do grupo a que o imóvel pertence.

    Equivale a uma tabela de consulta: dado o número de hóspedes (e
    opcionalmente o tipo de acomodação), devolve o preço típico. Grupos
    não vistos no treino recebem a mediana global.

    Args:
        df_treino: DataFrame de treino com as colunas de grupo e de preço.
        df_teste: DataFrame de teste com as mesmas colunas de grupo.
        colunas_grupo: colunas que definem o grupo.

    Returns:
        Vetor de previsões alinhado às linhas de `df_teste`.

    Raises:
        KeyError: se alguma coluna de grupo não existir nos DataFrames.
    """
    alvo = config.COLUNA_ALVO
    for coluna in colunas_grupo:
        if coluna not in df_treino.columns or coluna not in df_teste.columns:
            raise KeyError(f"Coluna de grupo ausente: {coluna}")

    medianas = df_treino.groupby(colunas_grupo)[alvo].median()
    global_ = float(df_treino[alvo].median())

    chaves = pd.MultiIndex.from_frame(df_teste[colunas_grupo]) \
        if len(colunas_grupo) > 1 else pd.Index(df_teste[colunas_grupo[0]])
    previsto = medianas.reindex(chaves).to_numpy(dtype=float)
    return np.where(np.isnan(previsto), global_, previsto)
