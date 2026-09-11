"""Testes do módulo de limpeza de dados."""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import cleaning


class TestConverterPreco(unittest.TestCase):
    """Conversão do preço textual para número (T2)."""

    def test_remove_cifrao_e_separador_de_milhar(self) -> None:
        entrada = pd.Series(["$354.00", "$1,234.50", "$99.99"])
        esperado = [354.00, 1234.50, 99.99]
        resultado = cleaning.converter_preco(entrada)
        self.assertEqual(list(resultado), esperado)

    def test_valor_invalido_vira_nulo(self) -> None:
        """Corner case: a base tem 8,4% de anúncios sem preço."""
        resultado = cleaning.converter_preco(pd.Series(["$100.00", None, "indefinido"]))
        self.assertEqual(resultado.isna().sum(), 2)


class TestExtrairBanheiros(unittest.TestCase):
    """Extração do número de banheiros do campo textual (T5)."""

    def test_extrai_inteiros_e_decimais(self) -> None:
        entrada = pd.Series(["1 bath", "2.5 baths", "3 shared baths"])
        self.assertEqual(list(cleaning.extrair_banheiros(entrada)), [1.0, 2.5, 3.0])

    def test_meio_banheiro_vale_zero_virgula_cinco(self) -> None:
        """Corner case: 'Half-bath' não tem dígito e exige regra própria."""
        resultado = cleaning.extrair_banheiros(pd.Series(["Half-bath"]))
        self.assertEqual(resultado.iloc[0], 0.5)


class TestFiltrarFaixaPreco(unittest.TestCase):
    """Remoção de nulos e outliers (T3 e T4)."""

    def setUp(self) -> None:
        """Prepara uma base com um outlier extremo e um valor ausente."""
        self.df = pd.DataFrame({"price": [150.0, 480.0, 574013.0, np.nan, 50.0]})

    def test_mantem_apenas_registros_na_faixa(self) -> None:
        resultado = cleaning.filtrar_faixa_preco(self.df, minimo=100.0, maximo=5000.0)
        self.assertEqual(len(resultado), 2)
        self.assertEqual(list(resultado["price"]), [150.0, 480.0])

    def test_faixa_invertida_levanta_erro(self) -> None:
        with self.assertRaises(ValueError):
            cleaning.filtrar_faixa_preco(self.df, minimo=5000.0, maximo=100.0)


class TestImputarPorMediana(unittest.TestCase):
    """Preenchimento de valores ausentes (T6)."""

    def test_substitui_nulos_pela_mediana(self) -> None:
        df = pd.DataFrame({"bedrooms": [1.0, 2.0, 3.0, np.nan]})
        resultado = cleaning.imputar_por_mediana(df, ["bedrooms"])
        self.assertEqual(resultado["bedrooms"].iloc[3], 2.0)
        self.assertEqual(resultado["bedrooms"].isna().sum(), 0)

    def test_coluna_inexistente_e_ignorada(self) -> None:
        df = pd.DataFrame({"bedrooms": [1.0, 2.0]})
        resultado = cleaning.imputar_por_mediana(df, ["coluna_que_nao_existe"])
        self.assertEqual(len(resultado), 2)


if __name__ == "__main__":
    unittest.main()
