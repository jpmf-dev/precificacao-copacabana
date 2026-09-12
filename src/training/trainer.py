"""Laço de treinamento do modelo.

Segue exatamente as cinco etapas vistas em aula: passagem direta, cálculo
da perda, zeragem dos gradientes, retropropagação e passo do otimizador.
O otimizador é construído antes do laço, nunca dentro dele.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src import config
from src.models.network import RegressorPreco, selecionar_dispositivo


@dataclass
class HistoricoTreino:
    """Registro das perdas ao longo das épocas.

    Attributes:
        epocas: número da época registrada.
        perda_treino: MAE no conjunto de treino.
        perda_teste: MAE no conjunto de teste.
    """

    epocas: list[int] = field(default_factory=list)
    perda_treino: list[float] = field(default_factory=list)
    perda_teste: list[float] = field(default_factory=list)


def fixar_semente(semente: int = config.SEMENTE) -> None:
    """Fixa as sementes aleatórias de PyTorch e NumPy.

    Sem isso os pesos iniciais mudam a cada execução e o RNF03 não se
    sustenta. É a única fonte de aleatoriedade do pipeline, já que a
    divisão treino/teste usa seu próprio gerador com a mesma semente.

    Args:
        semente: valor inteiro que controla a sequência pseudoaleatória.
    """
    torch.manual_seed(semente)
    np.random.seed(semente)


def treinar(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_teste: np.ndarray,
    y_teste: np.ndarray,
    epocas: int = config.EPOCAS,
    taxa_aprendizado: float = config.TAXA_APRENDIZADO,
    dim_oculta: int = config.NEURONIOS_OCULTOS,
    verbose: bool = True,
) -> tuple[RegressorPreco, HistoricoTreino]:
    """Treina a rede e acompanha o erro em treino e teste.

    Args:
        x_treino: atributos de treino já padronizados.
        y_treino: preços de treino.
        x_teste: atributos de teste já padronizados.
        y_teste: preços de teste.
        epocas: número de passagens completas pelos dados.
        taxa_aprendizado: tamanho do passo do otimizador.
        dim_oculta: número de neurônios da camada oculta. Exposto como
            parâmetro para permitir comparar configurações sem editar o
            módulo de configuração.
        verbose: imprime o erro a cada 10% das épocas quando verdadeiro.

    Returns:
        Tupla com o modelo treinado e o histórico de perdas.

    Raises:
        ValueError: se o número de épocas não for positivo.
    """
    if epocas <= 0:
        raise ValueError(f"Número de épocas deve ser positivo, recebido {epocas}.")

    fixar_semente()
    dispositivo = selecionar_dispositivo()

    xt = torch.tensor(x_treino, dtype=torch.float32, device=dispositivo)
    yt = torch.tensor(y_treino, dtype=torch.float32, device=dispositivo)
    xv = torch.tensor(x_teste, dtype=torch.float32, device=dispositivo)
    yv = torch.tensor(y_teste, dtype=torch.float32, device=dispositivo)

    modelo = RegressorPreco(
        dim_entrada=xt.shape[1], dim_oculta=dim_oculta
    ).to(dispositivo)

    funcao_perda = nn.L1Loss()
    otimizador = torch.optim.Adam(params=modelo.parameters(), lr=taxa_aprendizado)

    historico = HistoricoTreino()
    intervalo = max(1, epocas // 10)

    for epoca in range(epocas):
        modelo.train()
        previsto = modelo(xt)
        perda = funcao_perda(previsto, yt)

        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

        modelo.eval()
        with torch.inference_mode():
            perda_teste = funcao_perda(modelo(xv), yv)

        if epoca % intervalo == 0 or epoca == epocas - 1:
            historico.epocas.append(epoca)
            historico.perda_treino.append(float(perda))
            historico.perda_teste.append(float(perda_teste))
            if verbose:
                print(
                    f"Época {epoca:4d} | "
                    f"MAE treino: R$ {float(perda):7.2f} | "
                    f"MAE teste: R$ {float(perda_teste):7.2f}"
                )

    return modelo, historico


def prever(modelo: RegressorPreco, x: np.ndarray) -> np.ndarray:
    """Gera previsões para uma matriz de atributos.

    Args:
        modelo: rede já treinada.
        x: atributos padronizados com a mesma média e desvio do treino.

    Returns:
        Vetor de preços estimados, em reais.
    """
    dispositivo = next(modelo.parameters()).device
    tensor = torch.tensor(x, dtype=torch.float32, device=dispositivo)

    modelo.eval()
    with torch.inference_mode():
        saida = modelo(tensor)
    return saida.cpu().numpy()
