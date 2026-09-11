"""Testes dos baselines sem aprendizado de máquina."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation import baselines


class TestBaselineMedianaGlobal(unittest.TestCase):
    """Previsão constante pela mediana do treino."""

    def test_retorna_vetor_do_tamanho_do_teste(self) -> None:
        treino = np.array([100.0, 200.0, 300.0])
        teste = np.array([1.0, 2.0, 3.0, 4.0])
        resultado = baselines.baseline_mediana_global(treino, teste)
        self.assertEqual(resultado.shape, (4,))

    def test_todos_os_valores_sao_a_mediana(self) -> None:
        treino = np.array([100.0, 200.0, 300.0])
        resultado = baselines.baseline_mediana_global(treino, np.array([0.0, 0.0]))
        self.assertTrue(np.all(resultado == 200.0))


class TestBaselineMedianaPorGrupo(unittest.TestCase):
    """Tabela de consulta por características do imóvel."""

    def setUp(self) -> None:
        self.treino = pd.DataFrame(
            {"accommodates": [2, 2, 4, 4], "price": [300.0, 400.0, 800.0, 900.0]}
        )

    def test_usa_a_mediana_do_grupo_correspondente(self) -> None:
        teste = pd.DataFrame({"accommodates": [2, 4]})
        resultado = baselines.baseline_mediana_por_grupo(
            self.treino, teste, ["accommodates"]
        )
        np.testing.assert_array_equal(resultado, [350.0, 850.0])

    def test_grupo_nao_visto_recebe_mediana_global(self) -> None:
        """Corner case: capacidade ausente do treino não pode virar NaN."""
        teste = pd.DataFrame({"accommodates": [16]})
        resultado = baselines.baseline_mediana_por_grupo(
            self.treino, teste, ["accommodates"]
        )
        self.assertEqual(resultado[0], 600.0)
        self.assertFalse(np.isnan(resultado[0]))

    def test_coluna_de_grupo_ausente_levanta_erro(self) -> None:
        with self.assertRaises(KeyError):
            baselines.baseline_mediana_por_grupo(
                self.treino, pd.DataFrame({"x": [1]}), ["inexistente"]
            )


if __name__ == "__main__":
    unittest.main()
