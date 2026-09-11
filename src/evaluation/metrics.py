"""Métricas de avaliação do modelo.

Implementadas em NumPy para que possam ser usadas tanto sobre as saídas
do PyTorch quanto sobre os baselines sem aprendizado de máquina.
"""

import numpy as np

from src import config


def erro_absoluto_medio(real: np.ndarray, previsto: np.ndarray) -> float:
    """Calcula o MAE (Mean Absolute Error) em reais.

    O MAE é simétrico: pune igualmente superestimar e subestimar o preço.
    Isso reflete o SG1, em que o gestor considera custosos os dois erros —
    imóvel encalhado de um lado, receita perdida do outro.

    Args:
        real: valores observados.
        previsto: valores estimados.

    Returns:
        Erro absoluto médio.

    Raises:
        ValueError: se os vetores tiverem tamanhos diferentes ou forem vazios.
    """
    real, previsto = np.asarray(real, dtype=float), np.asarray(previsto, dtype=float)
    if real.shape != previsto.shape:
        raise ValueError(f"Shapes diferentes: {real.shape} e {previsto.shape}.")
    if real.size == 0:
        raise ValueError("Não é possível calcular MAE de vetores vazios.")
    return float(np.mean(np.abs(real - previsto)))


def proporcao_dentro_tolerancia(
    real: np.ndarray,
    previsto: np.ndarray,
    tolerancia: float = config.TOLERANCIA_RELATIVA,
) -> float:
    """Proporção de estimativas dentro de uma margem relativa do valor real.

    Complementa o MAE respondendo à pergunta do negócio: em quantos casos
    o preço sugerido é utilizável na prática? É a métrica do RNF02.

    Args:
        real: valores observados.
        previsto: valores estimados.
        tolerancia: margem relativa aceita (0,25 equivale a +/-25%).

    Returns:
        Proporção entre 0 e 1.

    Raises:
        ValueError: se os vetores tiverem tamanhos diferentes ou forem vazios.
        ValueError: se a tolerância não for positiva.
    """
    real, previsto = np.asarray(real, dtype=float), np.asarray(previsto, dtype=float)
    if real.shape != previsto.shape:
        raise ValueError(f"Shapes diferentes: {real.shape} e {previsto.shape}.")
    if real.size == 0:
        raise ValueError("Não é possível calcular a proporção de vetores vazios.")
    if tolerancia <= 0:
        raise ValueError(f"Tolerância deve ser positiva, recebido {tolerancia}.")

    return float(np.mean(np.abs(real - previsto) <= tolerancia * np.abs(real)))
