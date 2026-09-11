"""Persistência do modelo e inferência para novos imóveis.

Salva apenas o `state_dict`, e não o objeto inteiro: o modelo serializado
por inteiro fica acoplado à estrutura de diretórios e às classes do
projeto, quebrando após refatorações.

A média e o desvio da padronização são salvos junto com os pesos. Sem
eles, um imóvel novo seria padronizado com estatísticas diferentes das
do treino e a previsão sairia errada.
"""

from pathlib import Path

import numpy as np
import torch

from src import config
from src.models.network import RegressorPreco, selecionar_dispositivo


def salvar_modelo(
    modelo: RegressorPreco,
    media: np.ndarray,
    desvio: np.ndarray,
    nomes_atributos: list[str],
    caminho: Path | str = config.CAMINHO_MODELO,
) -> Path:
    """Grava pesos e parâmetros de padronização em disco.

    Args:
        modelo: rede treinada.
        media: médias usadas na padronização.
        desvio: desvios usados na padronização.
        nomes_atributos: nomes das colunas, na ordem da matriz de entrada.
        caminho: destino do arquivo `.pth`.

    Returns:
        Caminho do arquivo gravado.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        obj={
            "state_dict": modelo.state_dict(),
            "media": media,
            "desvio": desvio,
            "nomes_atributos": nomes_atributos,
            "dim_entrada": len(nomes_atributos),
        },
        f=caminho,
    )
    return caminho


def carregar_modelo(
    caminho: Path | str = config.CAMINHO_MODELO,
) -> tuple[RegressorPreco, np.ndarray, np.ndarray, list[str]]:
    """Reconstrói o modelo a partir do arquivo salvo.

    Atende ao RF08: a inferência carrega um modelo já treinado do disco em
    vez de retreinar, o que é o que viabiliza o tempo de resposta do RNF04.

    Args:
        caminho: caminho do arquivo `.pth`.

    Returns:
        Tupla (modelo, media, desvio, nomes_atributos).

    Raises:
        FileNotFoundError: se o arquivo não existir.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. Execute 'python main.py treinar' antes."
        )

    dispositivo = selecionar_dispositivo()
    pacote = torch.load(caminho, map_location=dispositivo, weights_only=False)

    modelo = RegressorPreco(dim_entrada=int(pacote["dim_entrada"]))
    modelo.load_state_dict(pacote["state_dict"])
    modelo.to(dispositivo)
    modelo.eval()

    return modelo, pacote["media"], pacote["desvio"], pacote["nomes_atributos"]


def estimar_preco(
    atributos: dict[str, float],
    caminho: Path | str = config.CAMINHO_MODELO,
) -> float:
    """Estima o preço-base da diária de um imóvel novo (RF07).

    Args:
        atributos: mapa de nome do atributo para valor. Atributos ausentes
            recebem zero, o que após a padronização corresponde à média.
        caminho: caminho do modelo salvo.

    Returns:
        Preço estimado em reais.
    """
    modelo, media, desvio, nomes = carregar_modelo(caminho)

    vetor = np.array([[float(atributos.get(nome, 0.0)) for nome in nomes]])
    padronizado = (vetor - media) / desvio

    dispositivo = next(modelo.parameters()).device
    tensor = torch.tensor(padronizado, dtype=torch.float32, device=dispositivo)
    with torch.inference_mode():
        saida = modelo(tensor)
    return float(saida.cpu().numpy()[0])
