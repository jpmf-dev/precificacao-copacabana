"""Ponto de entrada do sistema de precificação assistida.

Orquestra os módulos sem conter regra de negócio própria: cada etapa é
delegada ao pacote responsável. Isso mantém o `main` legível e permite
testar cada componente isoladamente.

Uso:
    python main.py treinar     # pipeline completo de ML + baselines
    python main.py sazonal     # constrói o índice de pressão de demanda
    python main.py estimar     # exemplo de inferência para um imóvel novo
"""

import sys
import time

import numpy as np

from src import config
from src.data import loader
from src.evaluation import baselines, metrics
from src.preprocessing import cleaning, features


def executar_treino() -> None:
    """Executa o pipeline completo: dados, treino, avaliação e persistência."""
    from src.inference import predictor
    from src.training import trainer

    print("=" * 68)
    print("PRECIFICAÇÃO ASSISTIDA — ALUGUEL POR TEMPORADA EM COPACABANA")
    print("=" * 68)

    print("\n[1/5] Carregando dados...")
    bruto = loader.carregar_listings()
    print(f"      Anúncios na cidade: {len(bruto):,}")

    bairro = loader.filtrar_bairro(bruto)
    print(f"      Anúncios em {config.BAIRRO}: {len(bairro):,}")

    print("\n[2/5] Pré-processando...")
    limpo = cleaning.preparar_anuncios(bairro)
    print(f"      Após limpeza e remoção de outliers: {len(limpo):,}")

    matriz, alvo, nomes = features.montar_matriz(limpo)
    print(f"      Matriz de atributos: {matriz.shape}")

    x_treino, x_teste, y_treino, y_teste = features.dividir_treino_teste(matriz, alvo)
    x_treino_pad, x_teste_pad, media, desvio = features.padronizar(x_treino, x_teste)
    print(f"      Treino: {len(x_treino):,} | Teste: {len(x_teste):,}")

    print("\n[3/5] Baselines sem aprendizado de máquina...")
    prev_global = baselines.baseline_mediana_global(y_treino, y_teste)
    mae_global = metrics.erro_absoluto_medio(y_teste, prev_global)
    dentro_global = metrics.proporcao_dentro_tolerancia(y_teste, prev_global)
    print(f"      Mediana global.......: MAE R$ {mae_global:7.2f} | "
          f"dentro de ±25%: {dentro_global:5.1%}")

    indices = features.dividir_treino_teste(
        np.arange(len(limpo)).reshape(-1, 1), alvo
    )
    idx_treino = indices[0].ravel()
    idx_teste = indices[1].ravel()
    df_treino = limpo.iloc[idx_treino]
    df_teste = limpo.iloc[idx_teste]

    prev_grupo = baselines.baseline_mediana_por_grupo(
        df_treino, df_teste, ["accommodates", "room_type"]
    )
    mae_grupo = metrics.erro_absoluto_medio(df_teste[config.COLUNA_ALVO].to_numpy(), prev_grupo)
    dentro_grupo = metrics.proporcao_dentro_tolerancia(
        df_teste[config.COLUNA_ALVO].to_numpy(), prev_grupo
    )
    print(f"      Mediana por grupo....: MAE R$ {mae_grupo:7.2f} | "
          f"dentro de ±25%: {dentro_grupo:5.1%}")

    print("\n[4/5] Treinando a rede neural...")
    inicio = time.perf_counter()
    modelo, _ = trainer.treinar(x_treino_pad, y_treino, x_teste_pad, y_teste)
    duracao = time.perf_counter() - inicio

    previsto = trainer.prever(modelo, x_teste_pad)
    mae_modelo = metrics.erro_absoluto_medio(y_teste, previsto)
    dentro_modelo = metrics.proporcao_dentro_tolerancia(y_teste, previsto)

    print("\n[5/5] Salvando o modelo...")
    caminho = predictor.salvar_modelo(modelo, media, desvio, nomes)
    print(f"      Gravado em {caminho}")

    print("\n" + "=" * 68)
    print("RESULTADO FINAL (conjunto de teste)")
    print("=" * 68)
    print(f"  Mediana global.........: MAE R$ {mae_global:7.2f} | {dentro_global:5.1%}")
    print(f"  Mediana por grupo......: MAE R$ {mae_grupo:7.2f} | {dentro_grupo:5.1%}")
    print(f"  Rede neural MLP........: MAE R$ {mae_modelo:7.2f} | {dentro_modelo:5.1%}")
    print(f"\n  Tempo de treino: {duracao:.1f}s")

    atende_rnf01 = mae_modelo < mae_grupo
    atende_rnf02 = dentro_modelo >= 0.60
    print(f"\n  RNF01 (superar baseline)........: {'ATENDIDO' if atende_rnf01 else 'NÃO ATENDIDO'}")
    print(f"  RNF02 (≥60% dentro de ±25%).....: {'ATENDIDO' if atende_rnf02 else 'NÃO ATENDIDO'}")
    print(f"  RNF05 (treino < 5 min em CPU)...: {'ATENDIDO' if duracao < 300 else 'NÃO ATENDIDO'}")


def executar_sazonal() -> None:
    """Constrói e grava a tabela de pressão de demanda (RF09)."""
    from src.seasonality import demand_index

    print("Carregando anúncios para obter os identificadores do bairro...")
    bairro = loader.filtrar_bairro(loader.carregar_listings())
    ids = set(bairro["id"])

    print(f"Carregando calendário ({len(ids):,} imóveis)...")
    calendario = loader.carregar_calendar(ids=ids)

    indice = demand_index.construir_indice(calendario, ids)
    config.CAMINHO_INDICE.parent.mkdir(parents=True, exist_ok=True)
    indice.to_csv(config.CAMINHO_INDICE, index=False)
    print(f"Índice gravado em {config.CAMINHO_INDICE} ({len(indice)} datas)")

    print("\nDatas com maior pressão de demanda:")
    for _, linha in indice.nlargest(10, "ipd").iterrows():
        print(f"  {linha['date'].date()}  IPD {linha['ipd']:.3f}  ({linha['faixa']})")


def executar_estimativa() -> None:
    """Demonstra a saída combinada ao gestor (RF10)."""
    import pandas as pd

    from src.inference import predictor
    from src.seasonality import demand_index

    exemplo = {
        "accommodates": 4.0,
        "bedrooms": 2.0,
        "beds": 2.0,
        "bathrooms": 1.0,
        "minimum_nights": 2.0,
        "availability_365": 180.0,
        "number_of_reviews": 25.0,
        "review_scores_rating": 4.8,
        "room_type_Entire home/apt": 1.0,
    }
    data = "2026-12-31"

    preco = predictor.estimar_preco(exemplo)
    print(f"Imóvel: {int(exemplo['accommodates'])} hóspedes, "
          f"{int(exemplo['bedrooms'])} quartos, {int(exemplo['bathrooms'])} banheiro")
    print(f"Preço-base estimado: R$ {preco:.2f}")

    if config.CAMINHO_INDICE.exists():
        indice = pd.read_csv(config.CAMINHO_INDICE, parse_dates=["date"])
        info = demand_index.consultar_data(indice, data)
        print(f"\nData consultada: {info['data']}")
        print(f"Pressão de demanda: {info['faixa']} (IPD {info['ipd']:.3f})")
        if info["faixa"] in ("alta", "crítica"):
            print("\n>> Recomendação: considerar ajuste para cima. "
                  "A decisão final é do gestor.")
    else:
        print("\n(Execute 'python main.py sazonal' para obter a pressão de demanda.)")


def main() -> None:
    """Roteia o comando informado na linha de comando."""
    comandos = {
        "treinar": executar_treino,
        "sazonal": executar_sazonal,
        "estimar": executar_estimativa,
    }
    escolhido = sys.argv[1] if len(sys.argv) > 1 else "treinar"

    if escolhido not in comandos:
        print(f"Comando desconhecido: {escolhido}")
        print(f"Disponíveis: {', '.join(comandos)}")
        sys.exit(1)

    comandos[escolhido]()


if __name__ == "__main__":
    main()
