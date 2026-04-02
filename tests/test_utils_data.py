import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import torch

from bioneuralnet.utils.data import (
    variance_summary,
    zero_summary,
    expression_summary,
    correlation_summary,
    nan_summary,
    sparse_filter,
    data_stats,
)
from bioneuralnet.utils.reproducibility import set_seed

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
            "Q": [1.0,  1.0,  1.0],
            "R": [0.0,  5.0,  0.0],
        })
        self.df_corr = pd.DataFrame({
            "U": [1.0, 2.0, 3.0],
            "V": [3.0, 2.0, 1.0],
            "W": [1.0, 0.0, 1.0],
        })

    def test_variance_summary_no_threshold(self):
        stats = variance_summary(self.df_var, var_threshold=None)
        self.assertAlmostEqual(stats["Variance Mean"], 4.0)
        self.assertAlmostEqual(stats["Variance Median"], 4.0)
        self.assertAlmostEqual(stats["Variance Min"], 4.0)
        self.assertAlmostEqual(stats["Variance Max"], 4.0)
        self.assertAlmostEqual(stats["Variance Std"], 0.0)
        self.assertNotIn("Number Of Low Variance Features", stats)

    def test_variance_summary_with_threshold(self):
        stats = variance_summary(self.df_var, var_threshold=5.0)
        self.assertIn("Number Of Low Variance Features", stats)
        self.assertEqual(stats["Number Of Low Variance Features"], 2)

    def test_zero_summary_no_threshold(self):
        stats = zero_summary(self.df_zero, zero_threshold=None)
        self.assertAlmostEqual(stats["Zero Mean"], 0.5)
        self.assertAlmostEqual(stats["Zero Median"], 0.5)
        self.assertAlmostEqual(stats["Zero Min"], 0.0)
        self.assertAlmostEqual(stats["Zero Max"], 1.0)
        expected_std = pd.Series([0.5, 0.0, 1.0]).std()
        self.assertAlmostEqual(stats["Zero Std"], expected_std)
        self.assertNotIn("Number Of High Zero Features", stats)

    def test_zero_summary_with_threshold(self):
        stats = zero_summary(self.df_zero, zero_threshold=0.5)
        self.assertIn("Number Of High Zero Features", stats)
        self.assertEqual(stats["Number Of High Zero Features"], 1)

    def test_expression_summary(self):
        stats = expression_summary(self.df_expr)
        mean_expr = self.df_expr.mean()
        self.assertAlmostEqual(stats["Expression Mean"],   float(mean_expr.mean()))
        self.assertAlmostEqual(stats["Expression Median"], float(mean_expr.median()))
        self.assertAlmostEqual(stats["Expression Min"],    float(mean_expr.min()))
        self.assertAlmostEqual(stats["Expression Max"],    float(mean_expr.max()))
        self.assertAlmostEqual(stats["Expression Std"],    float(mean_expr.std()))

    def test_correlation_summary(self):
        stats = correlation_summary(self.df_corr)
        corr_abs = self.df_corr.corr().abs()
        vals = corr_abs.to_numpy().copy()
        np.fill_diagonal(vals, 0.0)
        corr_abs = pd.DataFrame(vals, index=corr_abs.index, columns=corr_abs.columns)
        max_corr = corr_abs.max()
        self.assertAlmostEqual(stats["Max Corr Mean"],   float(max_corr.mean()))
        self.assertAlmostEqual(stats["Max Corr Median"], float(max_corr.median()))
        self.assertAlmostEqual(stats["Max Corr Min"],    float(max_corr.min()))
        self.assertAlmostEqual(stats["Max Corr Max"],    float(max_corr.max()))
        self.assertAlmostEqual(stats["Max Corr Std"],    float(max_corr.std()))

    def test_nan_summary_no_nans(self):
        pct = nan_summary(self.df_var, name="clean")
        self.assertAlmostEqual(pct, 0.0)

    def test_nan_summary_with_nans(self):
        df = pd.DataFrame({"x": [1.0, np.nan], "y": [np.nan, np.nan]})
        pct = nan_summary(df, name="sparse")
        self.assertAlmostEqual(pct, 75.0)

    def test_sparse_filter_drops_sparse_columns_and_rows(self):
        df = pd.DataFrame({
            "dense": [1.0, 2.0, 3.0, 4.0],
            "sparse": [np.nan, np.nan, np.nan, 1.0],
        })
        result = sparse_filter(df, missing_fraction=0.20)
        self.assertNotIn("sparse", result.columns)
        self.assertIn("dense", result.columns)

    def test_sparse_filter_clean_data_unchanged(self):
        result = sparse_filter(self.df_var, missing_fraction=0.20)
        self.assertEqual(result.shape, self.df_var.shape)

    @patch("bioneuralnet.utils.data.logger")
    def test_data_stats_logs_sections(self, mock_logger):
        data_stats(self.df_corr, name="TestDF", compute_correlation=True)
        all_args = [c[0][0] for c in mock_logger.info.call_args_list]
        output = "\n".join(all_args)
        self.assertIn("TestDF", output)
        self.assertIn("Variance", output)
        self.assertIn("Zero", output)
        self.assertIn("Expression", output)
        self.assertIn("Correlation", output)

    @patch("bioneuralnet.utils.data.logger")
    def test_data_stats_skips_correlation_when_false(self, mock_logger):
        data_stats(self.df_corr, name="TestDF", compute_correlation=False)
        all_args = [c[0][0] for c in mock_logger.info.call_args_list]
        output = "\n".join(all_args)
        self.assertIn("Skipped", output)

    def test_set_seed_reproducibility_numpy(self):
        set_seed(42)
        r1 = np.random.rand(5)
        set_seed(42)
        r2 = np.random.rand(5)
        np.testing.assert_array_equal(r1, r2)

    def test_set_seed_reproducibility_torch(self):
        set_seed(42)
        t1 = torch.rand(5)
        set_seed(42)
        t2 = torch.rand(5)
        self.assertTrue(torch.equal(t1, t2))

if __name__ == "__main__":
    unittest.main()
