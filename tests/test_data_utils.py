import unittest
import pandas as pd
import numpy as np
import io
import sys
from bioneuralnet.utils.data import variance_summary
from bioneuralnet.utils.data import zero_fraction_summary
from bioneuralnet.utils.data import expression_summary
from bioneuralnet.utils.data import correlation_summary
from bioneuralnet.utils.data import explore_data_stats

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

    def test_explore_data_stats_prints_all_sections(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        
        try:
            explore_data_stats(self.df_corr, name="TestDF")
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()

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
