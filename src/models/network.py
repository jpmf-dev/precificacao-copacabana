"""Definição da rede neural em PyTorch.

Responsabilidade única: descrever a arquitetura. O módulo não treina,
não avalia e não conhece o dataset — recebe apenas a dimensão de entrada.
"""

import torch
from torch import nn

from src import config


class RegressorPreco(nn.Module):
    """Perceptron multicamadas para estimar o preço-base da diária.

    Arquitetura: entrada -> camada oculta com ReLU -> saída escalar.

    A não linearidade é necessária porque a relação entre capacidade e
    preço em Copacabana é crescente e curva: o preço mediano vai de
    R$ 229 (1 hóspede) a R$ 438 (4) e R$ 1.065 (8). Uma regressão linear
    subestimaria os imóveis grandes.

    Attributes:
        rede: pilha sequencial de camadas.
    """

    def __init__(
        self,
        dim_entrada: int,
        dim_oculta: int = config.NEURONIOS_OCULTOS,
        dim_saida: int = 1,
    ) -> None:
        """Inicializa as camadas da rede.

        Args:
            dim_entrada: número de atributos de entrada.
            dim_oculta: número de neurônios da camada oculta.
            dim_saida: número de valores previstos (1 para regressão).

        Raises:
            ValueError: se alguma dimensão não for positiva.
        """
        super().__init__()
        if dim_entrada <= 0 or dim_oculta <= 0 or dim_saida <= 0:
            raise ValueError(
                "Dimensões devem ser positivas: "
                f"entrada={dim_entrada}, oculta={dim_oculta}, saida={dim_saida}."
            )

        self.rede = nn.Sequential(
            nn.Linear(in_features=dim_entrada, out_features=dim_oculta),
            nn.ReLU(),
            nn.Linear(in_features=dim_oculta, out_features=dim_saida),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem direta.

        Args:
            x: tensor de shape (n_amostras, dim_entrada).

        Returns:
            Tensor de shape (n_amostras,) com os preços estimados.
        """
        return self.rede(x).squeeze(-1)


def selecionar_dispositivo() -> torch.device:
    """Escolhe GPU quando disponível, com CPU como alternativa.

    Returns:
        Dispositivo onde modelo e tensores devem ser alocados.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
