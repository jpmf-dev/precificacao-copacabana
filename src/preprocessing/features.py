"""Construção da matriz de atributos e divisão dos dados (T8 a T10).

Este módulo é onde o NumPy faz o trabalho pesado: codificação one-hot,
padronização z-score e particionamento treino/teste operam sobre arrays,
não sobre DataFrames, para que a saída já esteja no formato consumido
pelo PyTorch.
"""

import numpy as np
import pandas as pd

from src import config


def codificar_categoricas(
    df: pd.DataFrame,
    colunas: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Aplica codificação one-hot às variáveis categóricas (transformação T8).

    Args:
        df: DataFrame limpo.
        colunas: colunas categóricas a codificar. Usa a configuração padrão
            quando omitido.

    Returns:
        Tupla com a matriz codificada (n_amostras, n_categorias) e a lista
        de nomes das colunas geradas.
    """
    colunas = colunas if colunas is not None else config.COLUNAS_CATEGORICAS
    presentes = [c for c in colunas if c in df.columns]
    if not presentes:
        return np.empty((len(df), 0), dtype=float), []

    codificado = pd.get_dummies(df[presentes], columns=presentes, dtype=float)
    return codificado.to_numpy(dtype=float), list(codificado.columns)


def montar_matriz(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Monta a matriz de atributos X e o vetor alvo y.

    Args:
        df: DataFrame limpo, já sem ausentes nas colunas numéricas.

    Returns:
        Tupla (X, y, nomes) onde X tem shape (n_amostras, n_atributos),
        y tem shape (n_amostras,) e nomes identifica cada coluna de X.

    Raises:
        KeyError: se alguma coluna numérica configurada estiver ausente.
    """
    faltando = [c for c in config.COLUNAS_NUMERICAS if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas numéricas ausentes: {faltando}")

    numericas = df[config.COLUNAS_NUMERICAS].to_numpy(dtype=float)
    categoricas, nomes_cat = codificar_categoricas(df)

    matriz = np.hstack([numericas, categoricas])
    alvo = df[config.COLUNA_ALVO].to_numpy(dtype=float)
    nomes = list(config.COLUNAS_NUMERICAS) + nomes_cat
    return matriz, alvo, nomes


def dividir_treino_teste(
    matriz: np.ndarray,
    alvo: np.ndarray,
    proporcao: float = config.PROPORCAO_TREINO,
    semente: int = config.SEMENTE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separa os dados em treino e teste com embaralhamento (transformação T10).

    O embaralhamento usa um gerador com semente fixa: a mesma semente
    sempre produz a mesma partição, o que sustenta o RNF03.

    Args:
        matriz: matriz de atributos.
        alvo: vetor de valores reais.
        proporcao: fração destinada ao treino, entre 0 e 1 exclusivos.
        semente: semente do gerador aleatório.

    Returns:
        Tupla (X_treino, X_teste, y_treino, y_teste).

    Raises:
        ValueError: se a proporção estiver fora de (0, 1) ou se matriz e
            alvo tiverem números de linhas diferentes.
    """
    if not 0.0 < proporcao < 1.0:
        raise ValueError(f"Proporção deve estar entre 0 e 1, recebido {proporcao}.")
    if len(matriz) != len(alvo):
        raise ValueError(
            f"Matriz e alvo com tamanhos diferentes: {len(matriz)} e {len(alvo)}."
        )

    gerador = np.random.default_rng(semente)
    indices = gerador.permutation(len(matriz))
    corte = int(proporcao * len(matriz))
    treino, teste = indices[:corte], indices[corte:]
    return matriz[treino], matriz[teste], alvo[treino], alvo[teste]


def padronizar(
    treino: np.ndarray,
    teste: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aplica padronização z-score às features (transformação T9).

    Média e desvio são calculados **apenas** no conjunto de treino e depois
    aplicados ao teste. Calcular no conjunto completo vazaria informação do
    teste para o treino e inflaria artificialmente o desempenho.

    Desvios nulos (colunas constantes) são substituídos por 1 para evitar
    divisão por zero.

    Args:
        treino: matriz de treino.
        teste: matriz de teste.

    Returns:
        Tupla (treino_padronizado, teste_padronizado, media, desvio).
    """
    media = treino.mean(axis=0)
    desvio = treino.std(axis=0)
    desvio = np.where(desvio == 0.0, 1.0, desvio)
    return (treino - media) / desvio, (teste - media) / desvio, media, desvio
