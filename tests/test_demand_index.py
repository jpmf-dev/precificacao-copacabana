"""Testes do componente determinístico de sazonalidade."""

import unittest

import pandas as pd

from src.seasonality import demand_index


class TestClassificar(unittest.TestCase):
    """Conversão do índice numérico em faixa."""

    def test_faixas_por_limite(self) -> None:
        self.assertEqual(demand_index.classificar(1.00), "normal")
        self.assertEqual(demand_index.classificar(1.10), "elevada")
        self.assertEqual(demand_index.classificar(1.15), "alta")
        self.assertEqual(demand_index.classificar(1.25), "crítica")


class TestNormalizarPorJanela(unittest.TestCase):
    """Correção do viés de horizonte pela janela móvel (T13)."""

    def test_serie_constante_resulta_em_razao_unitaria(self) -> None:
        serie = pd.Series([0.5] * 60)
        resultado = demand_index.normalizar_por_janela(serie, janela_dias=7)
        self.assertTrue((resultado.round(6) == 1.0).all())

    def test_tendencia_crescente_e_neutralizada(self) -> None:
        """Ocupação que só cresce com o horizonte é artefato, não demanda.

        Após a normalização, nenhuma data deve se destacar por estar
        distante no tempo.
        """
        serie = pd.Series([i / 100.0 for i in range(30, 90)])
        resultado = demand_index.normalizar_por_janela(serie, janela_dias=10)
        self.assertLess(resultado.max() - resultado.min(), 0.5)

    def test_pico_isolado_sobrevive_a_normalizacao(self) -> None:
        """O réveillon deve continuar visível depois da correção."""
        valores = [0.4] * 60
        valores[30] = 0.95
        resultado = demand_index.normalizar_por_janela(pd.Series(valores), 10)
        self.assertGreater(resultado.iloc[30], 2.0)

    def test_janela_nao_positiva_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            demand_index.normalizar_por_janela(pd.Series([1.0]), janela_dias=0)


class TestConstruirIndice(unittest.TestCase):
    """Construção da tabela de pressão de demanda."""

    def setUp(self) -> None:
        """Monta um calendário sintético de 40 dias e 4 imóveis.

        O dia 20 recebe mínimo de noites elevado, simulando um feriado.
        """
        datas = pd.date_range("2026-07-01", periods=40).strftime("%Y-%m-%d")
        linhas = []
        for i, data in enumerate(datas):
            for imovel in (1, 2, 3, 4):
                linhas.append(
                    {
                        "listing_id": imovel,
                        "date": data,
                        "available": "f" if i == 20 else "t",
                        "minimum_nights": 5 if i == 20 else 2,
                    }
                )
        self.calendario = pd.DataFrame(linhas)

    def test_gera_uma_linha_por_data(self) -> None:
        resultado = demand_index.construir_indice(self.calendario, {1, 2, 3, 4}, 7)
        self.assertEqual(len(resultado), 40)

    def test_data_de_feriado_recebe_indice_mais_alto(self) -> None:
        resultado = demand_index.construir_indice(self.calendario, {1, 2, 3, 4}, 7)
        pico = resultado.loc[resultado["date"] == pd.Timestamp("2026-07-21"), "ipd"]
        mediana = resultado["ipd"].median()
        self.assertGreater(pico.iloc[0], mediana)

    def test_ids_sem_correspondencia_levantam_erro(self) -> None:
        with self.assertRaises(ValueError):
            demand_index.construir_indice(self.calendario, {999}, 7)

    def test_consulta_de_data_fora_do_calendario_levanta_erro(self) -> None:
        indice = demand_index.construir_indice(self.calendario, {1, 2, 3, 4}, 7)
        with self.assertRaises(KeyError):
            demand_index.consultar_data(indice, "2030-01-01")

    def test_datas_com_cobertura_baixa_sao_descartadas(self) -> None:
        """Datas vistas por poucos imóveis geram falsos picos e saem fora.

        Simula o fim do horizonte: uma data extra observada por um único
        imóvel, com mínimo de noites alto.
        """
        extra = pd.DataFrame(
            [{
                "listing_id": 1,
                "date": "2026-08-25",
                "available": "f",
                "minimum_nights": 9,
            }]
        )
        calendario = pd.concat([self.calendario, extra], ignore_index=True)
        resultado = demand_index.construir_indice(calendario, {1, 2, 3, 4}, 7)
        self.assertNotIn(pd.Timestamp("2026-08-25"), list(resultado["date"]))

    def test_cobertura_invalida_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            demand_index.construir_indice(
                self.calendario, {1, 2, 3, 4}, 7, cobertura_minima=0.0
            )

    def test_execucoes_repetidas_produzem_indice_identico(self) -> None:
        """Evidência do RNF07: o componente é determinístico."""
        a = demand_index.construir_indice(self.calendario, {1, 2, 3, 4}, 7)
        b = demand_index.construir_indice(self.calendario, {1, 2, 3, 4}, 7)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
