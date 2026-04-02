import unittest
import warnings
import numpy as np
import pandas as pd

from bioneuralnet.utils.feature_selection import (
    variance_threshold,
    mad_filter,
    pca_loadings,
    laplacian_score,
    correlation_filter,
    importance_rf,
    top_anova_f_features,
)
from bioneuralnet.datasets import DatasetLoader

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]
        Y = example["Y"]

        self.X = pd.concat([X1, X2], axis=1)

        if isinstance(Y, pd.DataFrame):
            Y = Y.iloc[:, 0]
        self.y_cont = Y
        self.y_bin = (Y > float(np.median(Y.values))).astype(int)

        # small fixture for precise tests
        self.X_small = pd.DataFrame(
            {
                "high_var": [1.0, 10.0, 100.0, 1000.0],
                "low_var":  [1.0,  1.1,   1.0,    1.1],
                "zero_var": [5.0,  5.0,   5.0,    5.0],
            }
        )

        self.df_nan = pd.DataFrame(
            {
                "C1": [1.0, 2.0, np.nan, 4.0],
                "C2": [10.0, np.nan, 30.0, 40.0],
                "C3": [5.0, 5.0, 5.0, 5.0],
            }
        )

    def test_variance_threshold_selects_top_k(self):
        result = variance_threshold(self.X_small, k=2)
        self.assertEqual(result.shape[1], 2)
        # zero-variance column should be dropped first
        self.assertNotIn("zero_var", result.columns)

    def test_variance_threshold_k_larger_than_cols(self):
        # k > number of columns should return all columns (minus zero-var after clean)
        result = variance_threshold(self.X_small, k=100)
        self.assertLessEqual(result.shape[1], self.X_small.shape[1])

    def test_variance_threshold_on_example_data(self):
        X_sub = self.X.iloc[:, :50]
        result = variance_threshold(X_sub, k=10)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(result.shape[1], 10)
        self.assertEqual(result.shape[0], X_sub.shape[0])

    def test_mad_filter_reduces_columns(self):
        X_sub = self.X.iloc[:, :20]
        result = mad_filter(X_sub, n_keep=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(result.shape[1], 5)
        self.assertEqual(result.shape[0], X_sub.shape[0])

    def test_mad_filter_passthrough_when_n_keep_ge_cols(self):
        X_sub = self.X.iloc[:, :5]
        result = mad_filter(X_sub, n_keep=100)
        self.assertEqual(result.shape, X_sub.shape)

    def test_pca_loadings_reduces_columns(self):
        X_sub = self.X.iloc[:, :30]
        result = pca_loadings(X_sub, n_keep=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(result.shape[1], 5)
        self.assertEqual(result.shape[0], X_sub.shape[0])

    def test_pca_loadings_passthrough_when_n_keep_ge_cols(self):
        X_sub = self.X.iloc[:, :5]
        result = pca_loadings(X_sub, n_keep=100)
        self.assertEqual(result.shape, X_sub.shape)

    def test_laplacian_score_reduces_columns(self):
        X_sub = self.X.iloc[:, :20]
        result = laplacian_score(X_sub, n_keep=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(result.shape[1], 5)
        self.assertEqual(result.shape[0], X_sub.shape[0])

    def test_laplacian_score_passthrough_when_n_keep_ge_cols(self):
        X_sub = self.X.iloc[:, :5]
        result = laplacian_score(X_sub, n_keep=100)
        self.assertEqual(result.shape, X_sub.shape)

    def test_correlation_filter_unsupervised(self):
        X_sub = self.X.iloc[:40, :20]
        result = correlation_filter(X_sub, y=None, top_k=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 40)
        self.assertLessEqual(result.shape[1], 5)

    def test_correlation_filter_supervised(self):
        X_sub = self.X.iloc[:40, :20]
        y_sub = self.y_cont.iloc[:40]
        result = correlation_filter(X_sub, y=y_sub, top_k=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 40)
        self.assertLessEqual(result.shape[1], 5)

    def test_correlation_filter_bad_y_raises(self):
        X_sub = self.X.iloc[:5, :10]
        bad_y = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
        with self.assertRaises(ValueError):
            correlation_filter(X_sub, y=bad_y, top_k=3)

    def test_importance_rf_classification(self):
        X_sub = self.X.iloc[:40, :20]
        y_sub = self.y_bin.iloc[:40]
        result = importance_rf(X_sub, y_sub, top_k=5, seed=0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 40)
        self.assertLessEqual(result.shape[1], 5)

    def test_importance_rf_non_numeric_raises(self):
        X_sub = self.X.iloc[:10, :5].copy()
        X_sub["text_col"] = ["a"] * 10
        y_sub = self.y_bin.iloc[:10]
        with self.assertRaises(ValueError):
            importance_rf(X_sub, y_sub, top_k=3)

    def test_importance_rf_bad_y_raises(self):
        X_sub = self.X.iloc[:10, :5]
        bad_y = pd.DataFrame({"a": range(10), "b": range(10)})
        with self.assertRaises(ValueError):
            importance_rf(X_sub, bad_y, top_k=3)

    def test_top_anova_classification(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_sub = self.X.iloc[:40, :20]
            y_sub = self.y_bin.iloc[:40]
            result = top_anova_f_features(X_sub, y_sub, max_features=5, task="classification")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape[0], 40)
            self.assertLessEqual(result.shape[1], 5)

    def test_top_anova_regression(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_sub = self.X.iloc[:40, :20]
            y_sub = self.y_cont.iloc[:40]
            result = top_anova_f_features(X_sub, y_sub, max_features=5, task="regression")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape[0], 40)
            self.assertLessEqual(result.shape[1], 5)

    def test_top_anova_invalid_task_raises(self):
        X_sub = self.X.iloc[:40, :20]
        y_sub = self.y_cont.iloc[:40]
        with self.assertRaises(ValueError):
            top_anova_f_features(X_sub, y_sub, max_features=5, task="invalid")


if __name__ == "__main__":
    unittest.main()
