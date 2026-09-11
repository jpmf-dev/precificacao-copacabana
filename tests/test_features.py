"""Testes da construção de features e divisão dos dados."""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import features


class TestDividirTreinoTeste(unittest.TestCase):
    """Particionamento treino/teste (T10)."""

    def setUp(self) -> None:
        """Cria uma matriz de 100 amostras com 4 atributos."""
        self.matriz = np.arange(400, dtype=float).reshape(100, 4)
        self.alvo = np.arange(100, dtype=float)

    def test_proporcao_oitenta_vinte(self) -> None:
        xt, xv, yt, yv = features.dividir_treino_teste(self.matriz, self.alvo, 0.8)
        self.assertEqual(len(xt), 80)
        self.assertEqual(len(xv), 20)
        self.assertEqual(len(yt), 80)
        self.assertEqual(len(yv), 20)

    def test_mesma_semente_produz_mesma_particao(self) -> None:
        """Evidência do RNF03: duas execuções devem dar resultado idêntico."""
        a = features.dividir_treino_teste(self.matriz, self.alvo, 0.8, semente=42)
        b = features.dividir_treino_teste(self.matriz, self.alvo, 0.8, semente=42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[3], b[3])

    def test_sementes_diferentes_produzem_particoes_diferentes(self) -> None:
        a = features.dividir_treino_teste(self.matriz, self.alvo, 0.8, semente=42)
        b = features.dividir_treino_teste(self.matriz, self.alvo, 0.8, semente=7)
        self.assertFalse(np.array_equal(a[3], b[3]))

    def test_proporcao_invalida_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            features.dividir_treino_teste(self.matriz, self.alvo, 1.5)

    def test_tamanhos_incompativeis_levantam_erro(self) -> None:
        with self.assertRaises(ValueError):
            features.dividir_treino_teste(self.matriz, self.alvo[:50], 0.8)


class TestPadronizar(unittest.TestCase):
    """Padronização z-score (T9)."""

    def test_treino_fica_com_media_zero_e_desvio_um(self) -> None:
        treino = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        teste = np.array([[2.0, 200.0]])
        t_pad, _, _, _ = features.padronizar(treino, teste)
        np.testing.assert_allclose(t_pad.mean(axis=0), [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(t_pad.std(axis=0), [1.0, 1.0], atol=1e-9)

    def test_estatisticas_vem_do_treino(self) -> None:
        """O teste não pode influenciar média e desvio (evita vazamento)."""
        treino = np.array([[1.0], [2.0], [3.0]])
        teste = np.array([[1000.0]])
        _, _, media, desvio = features.padronizar(treino, teste)
        self.assertEqual(media[0], 2.0)
        self.assertAlmostEqual(desvio[0], np.std([1.0, 2.0, 3.0]))

    def test_coluna_constante_nao_divide_por_zero(self) -> None:
        """Corner case: desvio zero quebraria a divisão."""
        treino = np.array([[5.0], [5.0], [5.0]])
        teste = np.array([[5.0]])
        t_pad, v_pad, _, _ = features.padronizar(treino, teste)
        self.assertTrue(np.all(np.isfinite(t_pad)))
        self.assertTrue(np.all(np.isfinite(v_pad)))


class TestMontarMatriz(unittest.TestCase):
    """Montagem da matriz de atributos."""

    def test_coluna_ausente_levanta_erro(self) -> None:
        df = pd.DataFrame({"price": [100.0], "accommodates": [2.0]})
        with self.assertRaises(KeyError):
            features.montar_matriz(df)


if __name__ == "__main__":
    unittest.main()
