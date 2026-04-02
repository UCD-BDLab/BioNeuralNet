import unittest
import numpy as np
import pandas as pd

from bioneuralnet.utils.preprocess import (
    m_transform,
    impute_simple,
    impute_knn,
    normalize,
    clean_inf_nan,
    clean_internal,
    preprocess_clinical,
    prune_network,
    prune_network_by_quantile,
    network_remove_low_variance,
    network_remove_high_zero_fraction,
)
from bioneuralnet.datasets import DatasetLoader

class TestPreprocessFunctions(unittest.TestCase):
    def setUp(self):
        example = DatasetLoader("example")
        self.clinical = example["clinical"].copy()

        self.df_beta = pd.DataFrame(
            {
                "B1": [0.1, 0.5, 0.9],
                "B2": [0.0, 1.0, 0.5],
            }
        )

        # zero variance example
        self.df_nan = pd.DataFrame(
            {
                "C1": [1.0, 2.0, np.nan, 4.0],
                "C2": [10.0, np.nan, 30.0, 40.0],
                "C3": [5.0, 5.0, 5.0, 5.0],
            }
        )

        self.df_var = pd.DataFrame(
            {
                "A": [1.0, 3.0, 5.0],
                "B": [2.0, 4.0, 6.0],
            }
        )

        adj = np.array(
            [
                [1.0, 0.2, 0.0, 0.8],
                [0.2, 1.0, 0.1, 0.0],
                [0.0, 0.1, 1.0, 0.05],
                [0.8, 0.0, 0.05, 1.0],
            ],
            dtype=float,
        )
        self.adj_df = pd.DataFrame(
            adj, index=["a", "b", "c", "d"], columns=["a", "b", "c", "d"]
        )

    def test_m_transform_shape_preserved(self):
        result = m_transform(self.df_beta, eps=1e-6)
        self.assertEqual(result.shape, self.df_beta.shape)
        self.assertFalse(result.isnull().any().any())

    def test_m_transform_values(self):
        result = m_transform(self.df_beta, eps=1e-6)
        # B1=0.1 -> log2(0.1/0.9) around -3.17
        self.assertAlmostEqual(result.loc[0, "B1"], -3.169925, places=4)
        # B1=0.9 -> log2(0.9/0.1) around +3.17
        self.assertAlmostEqual(result.loc[2, "B1"],  3.169925, places=4)

    def test_impute_simple_mean(self):
        result = impute_simple(self.df_nan[["C1", "C2"]], method="mean")
        self.assertEqual(result.isna().sum().sum(), 0)
        self.assertAlmostEqual(result.loc[2, "C1"], (1.0 + 2.0 + 4.0) / 3, places=5)

    def test_impute_simple_median(self):
        result = impute_simple(self.df_nan[["C1", "C2"]], method="median")
        self.assertEqual(result.isna().sum().sum(), 0)
        self.assertAlmostEqual(result.loc[1, "C2"], 30.0, places=5)

    def test_impute_simple_zero(self):
        result = impute_simple(self.df_nan[["C1", "C2"]], method="zero")
        self.assertEqual(result.isna().sum().sum(), 0)
        self.assertEqual(result.loc[2, "C1"], 0.0)

    def test_impute_simple_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            impute_simple(self.df_nan, method="bad_method")

    def test_impute_knn_fills_nans(self):
        result = impute_knn(self.df_nan[["C1", "C2"]], n_neighbors=2)
        self.assertEqual(result.shape, self.df_nan[["C1", "C2"]].shape)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_impute_knn_no_nans_passthrough(self):
        df_clean = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = impute_knn(df_clean, n_neighbors=1)
        pd.testing.assert_frame_equal(result, df_clean)

    def test_impute_knn_non_numeric_raises(self):
        df_bad = pd.DataFrame({"x": [1.0, np.nan], "label": ["a", "b"]})
        with self.assertRaises(ValueError):
            impute_knn(df_bad)

    def test_normalize_standard(self):
        result = normalize(self.df_var, method="standard")
        self.assertAlmostEqual(result.mean().sum(), 0.0, places=5)

    def test_normalize_minmax(self):
        result = normalize(self.df_var, method="minmax")
        self.assertAlmostEqual(result.min().min(), 0.0, places=5)
        self.assertAlmostEqual(result.max().max(), 1.0, places=5)

    def test_normalize_log2(self):
        result = normalize(self.df_var, method="log2")
        self.assertAlmostEqual(result.loc[0, "A"], np.log2(2.0), places=5)

    def test_normalize_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            normalize(self.df_var, method="bad_method")

    # zero variance -> should be dropped
    def test_clean_inf_nan_removes_inf_and_nan(self):
        df = pd.DataFrame(
            {
                "x": [1.0, np.inf, 3.0, -np.inf],
                "y": [np.nan, 2.0, np.nan, 4.0],
                "z": [5.0, 5.0, 5.0, 5.0],
            }
        )
        cleaned = clean_inf_nan(df)
        self.assertFalse(cleaned.isin([np.inf, -np.inf]).any().any())
        self.assertFalse(cleaned.isna().any().any())
        self.assertNotIn("z", cleaned.columns)

    # 75 percent NaN and zero variance
    def test_clean_internal_drops_sparse_columns(self):
        df = pd.DataFrame(
            {
                "dense": [1.0, 2.0, 3.0, 4.0],
                "sparse": [np.nan, np.nan, np.nan, 1.0],
                "const":  [1.0, 1.0, 1.0, 1.0],
            }
        )
        result = clean_internal(df, nan_threshold=0.5)
        self.assertNotIn("sparse", result.columns)
        self.assertNotIn("const", result.columns)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_clean_internal_no_nans_passthrough(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = clean_internal(df, nan_threshold=0.5)
        self.assertEqual(result.shape, df.shape)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_preprocess_clinical_basic(self):
        result = preprocess_clinical(self.clinical)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], self.clinical.shape[0])
        self.assertGreater(result.shape[1], 0)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_preprocess_clinical_drop_columns(self):
        col_to_drop = self.clinical.columns[0]
        result = preprocess_clinical(self.clinical, drop_columns=[col_to_drop])
        self.assertNotIn(col_to_drop, result.columns)

    def test_preprocess_clinical_scale(self):
        result = preprocess_clinical(self.clinical, scale=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], self.clinical.shape[0])

    def test_preprocess_clinical_ordinal_mapping(self):
        df = pd.DataFrame(
            {
                "grade": ["low", "high", "low", "medium"],
                "score": [1.0, 2.0, 3.0, 4.0],
            }
        )
        mapping = {"ordinal_mappings": {"grade": {"low": 0, "medium": 1, "high": 2}}}
        result = preprocess_clinical(df, **mapping)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_prune_network_threshold(self):
        pruned = prune_network(self.adj_df, weight_threshold=0.15)
        self.assertIsInstance(pruned, pd.DataFrame)
        # all remaining nodes should be in the original index
        self.assertTrue(set(pruned.index).issubset(set(self.adj_df.index)))

    def test_prune_network_zero_threshold_keeps_all_connected(self):
        full = prune_network(self.adj_df, weight_threshold=0.0)
        self.assertEqual(set(full.index), set(self.adj_df.index))

    def test_prune_network_by_quantile(self):
        pruned = prune_network_by_quantile(self.adj_df, quantile=0.5)
        self.assertIsInstance(pruned, pd.DataFrame)
        self.assertTrue(set(pruned.index).issubset(set(self.adj_df.index)))

    def test_prune_network_by_quantile_empty_graph(self):
        empty = pd.DataFrame(
            np.zeros((3, 3)), index=["x", "y", "z"], columns=["x", "y", "z"]
        )
        result = prune_network_by_quantile(empty, quantile=0.5)
        self.assertIsInstance(result, pd.DataFrame)

    # zero variance -> should be removed
    def test_network_remove_low_variance(self):
        net = pd.DataFrame(
            {
                "n1": [1.0, 2.0, 1.0],
                "n2": [1.0, 1.0, 1.0],
                "n3": [3.0, 4.0, 5.0],
            },
            index=["n1", "n2", "n3"],
        )
        filtered = network_remove_low_variance(net, threshold=0.5)
        self.assertNotIn("n2", filtered.index)
        self.assertNotIn("n2", filtered.columns)

    # 2/3 zeros (c2)-> should be removed at threshold=0.5
    def test_network_remove_high_zero_fraction(self):
        net = pd.DataFrame(
            {
                "c1": [1.0, 0.5, 0.0],
                "c2": [1.0, 0.0, 0.0],
                "c3": [0.2, 0.2, 0.2],
            },
            index=["c1", "c2", "c3"],
        )
        filtered = network_remove_high_zero_fraction(net, threshold=0.5)
        self.assertNotIn("c2", filtered.index)
        self.assertNotIn("c2", filtered.columns)
        self.assertIn("c1", filtered.index)
        self.assertIn("c3", filtered.index)

if __name__ == "__main__":
    unittest.main()
