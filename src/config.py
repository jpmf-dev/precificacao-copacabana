"""Parâmetros centrais do projeto.

Manter as constantes em um único módulo evita que valores mágicos fiquem
espalhados pelo código e garante que treino, avaliação e testes usem
exatamente a mesma configuração (RNF03 — reprodutibilidade).
"""

from pathlib import Path
from typing import Final

RAIZ: Final[Path] = Path(__file__).resolve().parent.parent

CAMINHO_LISTINGS: Final[Path] = RAIZ / "data" / "raw" / "listings.csv.gz"
CAMINHO_CALENDAR: Final[Path] = RAIZ / "data" / "raw" / "calendar.csv.gz"
CAMINHO_MODELO: Final[Path] = RAIZ / "models" / "modelo_precificacao.pth"
CAMINHO_INDICE: Final[Path] = RAIZ / "data" / "processed" / "indice_demanda.csv"

BAIRRO: Final[str] = "Copacabana"

PRECO_MINIMO: Final[float] = 100.0
PRECO_MAXIMO: Final[float] = 5000.0

SEMENTE: Final[int] = 42
PROPORCAO_TREINO: Final[float] = 0.8

EPOCAS: Final[int] = 300
TAXA_APRENDIZADO: Final[float] = 0.01
NEURONIOS_OCULTOS: Final[int] = 32

TOLERANCIA_RELATIVA: Final[float] = 0.25
JANELA_SAZONAL_DIAS: Final[int] = 21
COBERTURA_MINIMA: Final[float] = 0.5
MAX_NOITES: Final[int] = 30

COLUNAS_NUMERICAS: Final[list[str]] = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
]

COLUNAS_CATEGORICAS: Final[list[str]] = ["room_type"]

COLUNA_ALVO: Final[str] = "price"
