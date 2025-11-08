import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import io
import sys
import logging
import torch
from bioneuralnet.utils.data import variance_summary
from bioneuralnet.utils.data import zero_fraction_summary
from bioneuralnet.utils.data import expression_summary
from bioneuralnet.utils.data import correlation_summary
from bioneuralnet.utils.data import explore_data_stats
from bioneuralnet.utils.data import impute_omics
from bioneuralnet.utils.data import impute_omics_knn
from bioneuralnet.utils.data import normalize_omics
from bioneuralnet.utils.data import beta_to_m
from bioneuralnet.utils.data import set_seed

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.df_var = pd.DataFrame({
            "A": [1.0, 3.0, 5.0],
            "B": [2.0, 4.0, 6.0],
        })
        self.df_zero = pd.DataFrame({
            "X": [0, 1, 2, 0],
            "Y": [1, 1, 1, 1],
            "Z": [0, 0, 0, 0],
        })
        self.df_expr = pd.DataFrame({
            "P": [10.0, 20.0, 30.0],
            "Q": [1.0, 1.0, 1.0],
            "R": [0.0, 5.0, 0.0],
        })
        self.df_corr = pd.DataFrame({
            "U": [1.0, 2.0, 3.0],
            "V": [3.0, 2.0, 1.0],
            "W": [1.0, 0.0, 1.0],
        })
        self.df_nan = pd.DataFrame({
            "C1": [1.0, 2.0, np.nan, 4.0],
            "C2": [10.0, np.nan, 30.0, 40.0],
            "C3": [5.0, 5.0, 5.0, 5.0],
        })
        self.df_beta = pd.DataFrame({
            "B1": [0.1, 0.5, 0.9],
            "B2": [0.0, 1.0, 0.5],
        })

        self.mock_logger = logging.getLogger('test_logger')
        self.mock_logger.setLevel(logging.INFO)
        self.mock_stream = io.StringIO()
        self.mock_handler = logging.StreamHandler(self.mock_stream)
        self.mock_logger.addHandler(self.mock_handler)

    def test_variance_summary_no_threshold(self):
        stats = variance_summary(self.df_var, low_var_threshold=None)

        self.assertAlmostEqual(stats["variance_mean"], 4.0)
        self.assertAlmostEqual(stats["variance_median"], 4.0)
        self.assertAlmostEqual(stats["variance_min"], 4.0)
        self.assertAlmostEqual(stats["variance_max"], 4.0)
        self.assertAlmostEqual(stats["variance_std"], 0.0)

        self.assertNotIn("num_low_variance_features", stats)

    def test_variance_summary_with_threshold(self):
        stats = variance_summary(self.df_var, low_var_threshold=5.0)

        self.assertIn("num_low_variance_features", stats)
        self.assertEqual(stats["num_low_variance_features"], 2)

    def test_zero_fraction_summary_no_threshold(self):
        stats = zero_fraction_summary(self.df_zero, high_zero_threshold=None)

        self.assertAlmostEqual(stats["zero_fraction_mean"], 0.5)
        self.assertAlmostEqual(stats["zero_fraction_median"], 0.5)
        self.assertAlmostEqual(stats["zero_fraction_min"], 0.0)
        self.assertAlmostEqual(stats["zero_fraction_max"], 1.0)

        expected_std = pd.Series([0.5, 0.0, 1.0]).std()

        self.assertAlmostEqual(stats["zero_fraction_std"], expected_std)
        self.assertNotIn("num_high_zero_features", stats)

    def test_zero_fraction_summary_with_threshold(self):
        stats = zero_fraction_summary(self.df_zero, high_zero_threshold=0.5)

        self.assertIn("num_high_zero_features", stats)
        self.assertEqual(stats["num_high_zero_features"], 1)

    def test_expression_summary(self):
        stats = expression_summary(self.df_expr)
        mean_exprs = pd.Series({"P": 20.0, "Q": 1.0, "R": (0.0 + 5.0 + 0.0) / 3})

        self.assertAlmostEqual(stats["expression_mean"], mean_exprs.mean())
        self.assertAlmostEqual(stats["expression_median"], mean_exprs.median())
        self.assertAlmostEqual(stats["expression_min"], mean_exprs.min())
        self.assertAlmostEqual(stats["expression_max"], mean_exprs.max())
        self.assertAlmostEqual(stats["expression_std"], mean_exprs.std())

    def test_correlation_summary(self):
        stats = correlation_summary(self.df_corr)
        corr_abs = self.df_corr.corr().abs()

        np.fill_diagonal(corr_abs.values, 0.0)
        max_corr = corr_abs.max()

        self.assertAlmostEqual(max_corr["U"], 1.0)
        self.assertAlmostEqual(max_corr["V"], 1.0)

        self.assertAlmostEqual(stats["max_corr_mean"], max_corr.mean())
        self.assertAlmostEqual(stats["max_corr_median"], max_corr.median())
        self.assertAlmostEqual(stats["max_corr_min"], max_corr.min())
        self.assertAlmostEqual(stats["max_corr_max"], max_corr.max())
        self.assertAlmostEqual(stats["max_corr_std"], max_corr.std())

    def test_impute_omics_mean(self):
        df_imputed = impute_omics(self.df_nan, method="mean")
        self.assertAlmostEqual(df_imputed.loc[2, "C1"], 2.3333333333333335)
        self.assertAlmostEqual(df_imputed.loc[1, "C2"], 26.666666666666668)
        self.assertEqual(df_imputed.isna().sum().sum(), 0)

    def test_impute_omics_median(self):
        df_imputed = impute_omics(self.df_nan, method="median")
        self.assertAlmostEqual(df_imputed.loc[1, "C2"], 30.0)
        self.assertEqual(df_imputed.isna().sum().sum(), 0)

    def test_impute_omics_knn(self):
        df_imputed = impute_omics_knn(self.df_nan, n_neighbors=2)
        self.assertEqual(df_imputed.shape, self.df_nan.shape)
        self.assertEqual(df_imputed.isna().sum().sum(), 0)
        self.assertNotAlmostEqual(df_imputed.loc[1, "C2"], 26.666666666666668)

    def test_normalize_omics_standard(self):
        df_normalized = normalize_omics(self.df_nan.dropna(), method="standard")
        self.assertAlmostEqual(df_normalized.mean().sum(), 0.0, places=5)
        expected_std_sum = np.sqrt(2) * 2
        self.assertAlmostEqual(df_normalized.std().sum(), expected_std_sum, places=5)

    def test_normalize_omics_log2(self):
        df_log = normalize_omics(self.df_var, method="log2")
        self.assertAlmostEqual(df_log.loc[0, "A"], 1.0)
        self.assertAlmostEqual(df_log.loc[0, "B"], 1.5849625)

    def test_beta_to_m_conversion(self):
        df_m_values = beta_to_m(self.df_beta, eps=1e-6)
        self.assertAlmostEqual(df_m_values.loc[0, "B1"], -3.169925)
        self.assertAlmostEqual(df_m_values.loc[2, "B2"], 0.0)
        self.assertAlmostEqual(df_m_values.loc[0, "B2"], -19.931567126628412)

    def test_set_seed_reproducibility(self):
        test_seed = 177

        set_seed(test_seed)
        result1 = np.random.rand(5)

        set_seed(test_seed)
        result2 = np.random.rand(5)

        np.testing.assert_array_equal(result1, result2)
        set_seed(test_seed + 1)
        tensor1 = torch.rand(5)

        set_seed(test_seed + 1)
        tensor2 = torch.rand(5)

        self.assertTrue(torch.equal(tensor1, tensor2))

    @patch('bioneuralnet.utils.data.logger')
    def test_explore_data_stats_logs_all_sections(self, mock_logger):
        mock_logger.addHandler(self.mock_handler)
        explore_data_stats(self.df_corr, name="TestDF")

        all_call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        output = "\n".join(all_call_args)

        self.assertIn("Statistics for TestDF:", output)
        self.assertIn("Variance Summary:", output)
        self.assertIn("Zero Fraction Summary:", output)
        self.assertIn("Expression Summary:", output)
        self.assertIn("Correlation Summary:", output)
        self.assertIn("variance_mean", output)
        self.assertIn("zero_fraction_mean", output)
        self.assertIn("expression_mean", output)
        self.assertIn("max_corr_mean", output)

if __name__ == "__main__":
    unittest.main()
