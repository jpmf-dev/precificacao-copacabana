"""Testes da rede neural e da persistência do modelo.

Estes testes dependem do PyTorch. Quando a biblioteca não está instalada,
a classe inteira é ignorada em vez de quebrar a suíte — assim o restante
dos testes continua executável em ambientes sem torch.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from src.inference import predictor
    from src.models.network import RegressorPreco

    TORCH_DISPONIVEL = True
except ImportError:
    TORCH_DISPONIVEL = False


@unittest.skipUnless(TORCH_DISPONIVEL, "PyTorch não instalado neste ambiente")
class TestRegressorPreco(unittest.TestCase):
    """Arquitetura e formato das saídas da rede."""

    def setUp(self) -> None:
        """Cria uma rede com 9 atributos de entrada."""
        torch.manual_seed(42)
        self.dim_entrada = 9
        self.modelo = RegressorPreco(dim_entrada=self.dim_entrada, dim_oculta=16)

    def test_saida_tem_uma_previsao_por_amostra(self) -> None:
        """Shape de saída deve ser (n_amostras,), não (n_amostras, 1)."""
        entrada = torch.randn(32, self.dim_entrada)
        saida = self.modelo(entrada)
        self.assertEqual(saida.shape, (32,))

    def test_lote_unitario_funciona(self) -> None:
        """Corner case: inferência de um único imóvel (RF07)."""
        saida = self.modelo(torch.randn(1, self.dim_entrada))
        self.assertEqual(saida.shape, (1,))

    def test_dimensao_invalida_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            RegressorPreco(dim_entrada=0)

    def test_mesma_semente_gera_mesmos_pesos(self) -> None:
        """Evidência do RNF03 no nível da inicialização."""
        torch.manual_seed(42)
        a = RegressorPreco(dim_entrada=5, dim_oculta=8)
        torch.manual_seed(42)
        b = RegressorPreco(dim_entrada=5, dim_oculta=8)
        for p_a, p_b in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p_a, p_b))

    def test_modelo_possui_parametros_treinaveis(self) -> None:
        treinaveis = [p for p in self.modelo.parameters() if p.requires_grad]
        self.assertGreater(len(treinaveis), 0)


@unittest.skipUnless(TORCH_DISPONIVEL, "PyTorch não instalado neste ambiente")
class TestPersistencia(unittest.TestCase):
    """Salvamento e carregamento do modelo (RF08)."""

    def setUp(self) -> None:
        """Prepara um modelo e um diretório temporário."""
        torch.manual_seed(42)
        self.nomes = [f"atributo_{i}" for i in range(5)]
        self.modelo = RegressorPreco(dim_entrada=5, dim_oculta=8)
        self.media = np.zeros(5)
        self.desvio = np.ones(5)
        self.tmp = tempfile.TemporaryDirectory()
        self.caminho = Path(self.tmp.name) / "modelo.pth"

    def tearDown(self) -> None:
        """Remove o diretório temporário."""
        self.tmp.cleanup()

    def test_arquivo_e_criado(self) -> None:
        predictor.salvar_modelo(
            self.modelo, self.media, self.desvio, self.nomes, self.caminho
        )
        self.assertTrue(self.caminho.exists())

    def test_modelo_carregado_preve_igual_ao_original(self) -> None:
        """O ciclo salvar/carregar não pode alterar as previsões."""
        entrada = torch.randn(10, 5)
        self.modelo.eval()
        with torch.inference_mode():
            antes = self.modelo(entrada)

        predictor.salvar_modelo(
            self.modelo, self.media, self.desvio, self.nomes, self.caminho
        )
        carregado, _, _, _ = predictor.carregar_modelo(self.caminho)
        with torch.inference_mode():
            depois = carregado(entrada)

        self.assertTrue(torch.allclose(antes, depois))

    def test_estatisticas_de_padronizacao_sao_preservadas(self) -> None:
        """Sem média e desvio, a inferência de um imóvel novo sairia errada."""
        media = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        desvio = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        predictor.salvar_modelo(self.modelo, media, desvio, self.nomes, self.caminho)
        _, m, d, nomes = predictor.carregar_modelo(self.caminho)
        np.testing.assert_array_equal(m, media)
        np.testing.assert_array_equal(d, desvio)
        self.assertEqual(nomes, self.nomes)

    def test_carregar_arquivo_inexistente_levanta_erro(self) -> None:
        with self.assertRaises(FileNotFoundError):
            predictor.carregar_modelo(Path(self.tmp.name) / "nao_existe.pth")


if __name__ == "__main__":
    unittest.main()
