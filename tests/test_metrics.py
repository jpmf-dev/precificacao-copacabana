"""Testes das métricas de avaliação."""

import unittest

import numpy as np

from src.evaluation import metrics


class TestErroAbsolutoMedio(unittest.TestCase):
    """Cálculo do MAE."""

    def test_previsao_perfeita_tem_erro_zero(self) -> None:
        valores = np.array([100.0, 200.0, 300.0])
        self.assertEqual(metrics.erro_absoluto_medio(valores, valores), 0.0)

    def test_calcula_media_dos_desvios_absolutos(self) -> None:
        real = np.array([100.0, 200.0])
        previsto = np.array([150.0, 100.0])
        self.assertEqual(metrics.erro_absoluto_medio(real, previsto), 75.0)

    def test_erro_e_simetrico(self) -> None:
        """Superestimar e subestimar em R$ 50 devem custar o mesmo (SG1)."""
        real = np.array([500.0])
        acima = metrics.erro_absoluto_medio(real, np.array([550.0]))
        abaixo = metrics.erro_absoluto_medio(real, np.array([450.0]))
        self.assertEqual(acima, abaixo)

    def test_vetor_vazio_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            metrics.erro_absoluto_medio(np.array([]), np.array([]))

    def test_shapes_diferentes_levantam_erro(self) -> None:
        with self.assertRaises(ValueError):
            metrics.erro_absoluto_medio(np.array([1.0, 2.0]), np.array([1.0]))


class TestProporcaoDentroTolerancia(unittest.TestCase):
    """Proporção de acertos dentro da margem relativa (RNF02)."""

    def test_conta_apenas_dentro_da_margem(self) -> None:
        real = np.array([100.0, 100.0, 100.0, 100.0])
        previsto = np.array([110.0, 130.0, 90.0, 70.0])
        self.assertEqual(metrics.proporcao_dentro_tolerancia(real, previsto, 0.25), 0.5)

    def test_limite_exato_conta_como_dentro(self) -> None:
        """Corner case: erro de exatamente 25% deve ser aceito."""
        real = np.array([100.0])
        previsto = np.array([125.0])
        self.assertEqual(metrics.proporcao_dentro_tolerancia(real, previsto, 0.25), 1.0)

    def test_tolerancia_nao_positiva_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            metrics.proporcao_dentro_tolerancia(np.array([1.0]), np.array([1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
