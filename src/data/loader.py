"""Carregamento das bases brutas do Inside Airbnb.

Responsabilidade única: ler arquivos do disco e devolver DataFrames.
Nenhuma limpeza ou transformação acontece aqui — isso é responsabilidade
do pacote `preprocessing`. Essa separação permite testar o carregamento
independentemente das regras de negócio (RNF06).
"""

from pathlib import Path

import pandas as pd

from src import config


def carregar_listings(caminho: Path | str = config.CAMINHO_LISTINGS) -> pd.DataFrame:
    """Lê o arquivo de anúncios do Inside Airbnb.

    Args:
        caminho: caminho do arquivo `listings.csv.gz`.

    Returns:
        DataFrame com todos os anúncios da cidade, sem filtros.

    Raises:
        FileNotFoundError: se o arquivo não existir no caminho informado.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de anúncios não encontrado em {caminho}. "
            "Baixe listings.csv.gz em https://insideairbnb.com/get-the-data "
            "e coloque em data/raw/."
        )
    return pd.read_csv(caminho, compression="gzip", low_memory=False)


def carregar_calendar(
    caminho: Path | str = config.CAMINHO_CALENDAR,
    ids: set[int] | None = None,
    tamanho_bloco: int = 2_000_000,
) -> pd.DataFrame:
    """Lê o calendário diário em blocos, filtrando durante a leitura.

    O arquivo tem cerca de 17,8 milhões de linhas para o Rio de Janeiro.
    Quando `ids` é informado, cada bloco é filtrado antes de ser acumulado:
    só as linhas dos imóveis de interesse chegam à memória. Sem isso, o
    ganho da leitura em blocos se perderia na concatenação final.

    Args:
        caminho: caminho do arquivo `calendar.csv.gz`.
        ids: identificadores dos imóveis a manter. Mantém todos se omitido.
        tamanho_bloco: número de linhas lidas por bloco.

    Returns:
        DataFrame com as colunas listing_id, date, available e minimum_nights.

    Raises:
        FileNotFoundError: se o arquivo não existir no caminho informado.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de calendário não encontrado em {caminho}. "
            "Baixe calendar.csv.gz em https://insideairbnb.com/get-the-data "
            "e coloque em data/raw/."
        )
    colunas = ["listing_id", "date", "available", "minimum_nights"]
    blocos = pd.read_csv(
        caminho,
        compression="gzip",
        usecols=colunas,
        chunksize=tamanho_bloco,
    )
    selecionados = [
        bloco if ids is None else bloco[bloco["listing_id"].isin(ids)]
        for bloco in blocos
    ]
    return pd.concat(selecionados, ignore_index=True)


def filtrar_bairro(df: pd.DataFrame, bairro: str = config.BAIRRO) -> pd.DataFrame:
    """Seleciona apenas os anúncios de um bairro (transformação T1).

    Args:
        df: DataFrame de anúncios com a coluna `neighbourhood_cleansed`.
        bairro: nome do bairro exatamente como aparece na base.

    Returns:
        Cópia do DataFrame contendo apenas as linhas do bairro.

    Raises:
        KeyError: se a coluna de bairro não existir no DataFrame.
        ValueError: se nenhum anúncio for encontrado para o bairro.
    """
    if "neighbourhood_cleansed" not in df.columns:
        raise KeyError("Coluna 'neighbourhood_cleansed' ausente no DataFrame.")

    filtrado = df[df["neighbourhood_cleansed"] == bairro].copy()
    if filtrado.empty:
        raise ValueError(f"Nenhum anúncio encontrado para o bairro '{bairro}'.")
    return filtrado
