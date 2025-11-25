import unittest
import pandas as pd
import numpy as np
import warnings

from bioneuralnet.utils.preprocess import preprocess_clinical
from bioneuralnet.utils.preprocess import clean_inf_nan
from bioneuralnet.utils.preprocess import select_top_k_variance
from bioneuralnet.utils.preprocess import select_top_k_correlation
from bioneuralnet.utils.preprocess import select_top_randomforest
from bioneuralnet.utils.preprocess import top_anova_f_features
from bioneuralnet.utils.preprocess import prune_network
from bioneuralnet.utils.preprocess import prune_network_by_quantile
from bioneuralnet.utils.preprocess import network_remove_low_variance
from bioneuralnet.utils.preprocess import network_remove_high_zero_fraction
from bioneuralnet.utils.preprocess import impute_omics
from bioneuralnet.utils.preprocess import impute_omics_knn
from bioneuralnet.utils.preprocess import normalize_omics
from bioneuralnet.utils.preprocess import beta_to_m
from bioneuralnet.datasets import DatasetLoader


class TestPreprocessFunctions(unittest.TestCase):
    def setUp(self):
        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]
        Y = example["Y"]
        clinical = example["clinical"]

        self.omics_example = pd.concat([X1, X2], axis=1)

        if isinstance(Y, pd.DataFrame):
            self.y_example = Y.iloc[:, 0]
        else:
            self.y_example = Y

        y_array = self.y_example.values
        median_val = float(np.median(y_array))
        self.y_binary = (self.y_example > median_val).astype(int)

        self.clinical = clinical.copy()
        self.clinical["ignore"] = 1

        self.X_small = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [1.0, 1.0, 1.0, 1.0],
                "f3": [4.0, 3.0, 2.0, 1.0],
                "f4": [10.0, 20.0, 30.0, 40.0],
            },
            index=["i1", "i2", "i3", "i4"],
        )

        self.df_var = pd.DataFrame(
            {
                "A": [1.0, 3.0, 5.0],
                "B": [2.0, 4.0, 6.0],
            }
        )

        self.df_nan = pd.DataFrame(
            {
                "C1": [1.0, 2.0, np.nan, 4.0],
                "C2": [10.0, np.nan, 30.0, 40.0],
                "C3": [5.0, 5.0, 5.0, 5.0],
            }
        )

        self.df_beta = pd.DataFrame(
            {
                "B1": [0.1, 0.5, 0.9],
                "B2": [0.0, 1.0, 0.5],
            }
        )

        self.y_reg = pd.Series([0.1, 0.4, 0.5, 0.8], index=self.X_small.index)

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

    def test_clean_inf_nan_replaces_and_drops(self):
        df = pd.DataFrame(
            {
                "x": [1.0, np.inf, 3.0, -np.inf],
                "y": [np.nan, 2.0, np.nan, 4.0],
                "z": [5.0, 5.0, 5.0, 5.0],
            }
        )
        cleaned = clean_inf_nan(df)

        self.assertTrue(np.allclose(cleaned["x"].values, [1.0, 2.0, 3.0, 2.0]))
        self.assertTrue(np.allclose(cleaned["y"].values, [3.0, 2.0, 3.0, 4.0]))
        self.assertNotIn("z", cleaned.columns)
        self.assertFalse(cleaned.isin([np.inf, -np.inf]).any().any())
        self.assertFalse(cleaned.isna().any().any())

    def test_preprocess_clinical_basic(self):
        result = preprocess_clinical(
            X=self.clinical,
            top_k=3,
            scale=False,
            ignore_columns=["ignore"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], self.clinical.shape[0])
        self.assertNotIn("ignore", result.columns)
        self.assertGreater(result.shape[1], 0)
        self.assertLessEqual(result.shape[1], 3)

    def test_preprocess_clinical_errors(self):
        with self.assertRaises(TypeError):
            preprocess_clinical(
                self.clinical.iloc[:2, :],
                top_k=1,
                nan_threshold="bad",
            )
        with self.assertRaises(KeyError):
            preprocess_clinical(self.clinical, ignore_columns=["nonexistent"])

    def test_select_top_k_variance(self):
        X_sub = self.omics_example.iloc[:40, :20]
        top5 = select_top_k_variance(X_sub, k=5)
        self.assertEqual(top5.shape[0], 40)
        self.assertLessEqual(top5.shape[1], 5)
        self.assertTrue((top5.var(axis=0) > 0).all())

    def test_select_top_k_correlation_unsupervised(self):
        X_sub = self.omics_example.iloc[:40, :20]
        unsup = select_top_k_correlation(X_sub, y=None, top_k=5)
        self.assertEqual(unsup.shape[0], 40)
        self.assertLessEqual(unsup.shape[1], 5)
        self.assertTrue((unsup.var(axis=0) > 0).all())

    def test_select_top_k_correlation_supervised(self):
        X_sub = self.omics_example.iloc[:40, :20]
        y_sub = self.y_example.iloc[:40]

        sup = select_top_k_correlation(X_sub, y=y_sub, top_k=5)
        self.assertEqual(sup.shape[0], 40)
        self.assertLessEqual(sup.shape[1], 5)

        bad_y = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with self.assertRaises(ValueError):
            select_top_k_correlation(X_sub.iloc[:2, :], bad_y, top_k=1)

    def test_select_top_randomforest(self):
        X_sub = self.omics_example.iloc[:40, :20]
        y_sub = self.y_example.iloc[:40]

        top_rf = select_top_randomforest(X_sub, y_sub, top_k=5, seed=0)
        self.assertEqual(top_rf.shape[0], 40)
        self.assertLessEqual(top_rf.shape[1], 5)

        X_mixed = X_sub.copy()
        str_list = []
        i = 0
        n_rows = X_sub.shape[0]
        base_strings = ["x", "y", "z", "w"]
        while len(str_list) < n_rows:
            str_list.append(base_strings[i % len(base_strings)])
            i += 1
        X_mixed["non_numeric"] = str_list

        with self.assertRaises(ValueError):
            select_top_randomforest(X_mixed, y_sub, top_k=1)

    def test_top_anova_f_features_classification(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            X_clf = self.omics_example.iloc[:40, :20]
            y_bin = self.y_binary.iloc[:40]

            top_feats = top_anova_f_features(
                X_clf,
                y_bin,
                max_features=5,
                alpha=0.05,
                task="classification",
            )
            self.assertEqual(top_feats.shape[0], 40)
            self.assertLessEqual(top_feats.shape[1], 5)

            X_reg = self.omics_example.iloc[:40, :20]
            y_cont = self.y_example.iloc[:40]
            top_feats_reg = top_anova_f_features(
                X_reg,
                y_cont,
                max_features=5,
                alpha=0.05,
                task="regression",
            )
            self.assertEqual(top_feats_reg.shape[0], 40)
            self.assertLessEqual(top_feats_reg.shape[1], 5)

            with self.assertRaises(ValueError):
                top_anova_f_features(
                    X_reg,
                    y_cont,
                    max_features=1,
                    alpha=0.05,
                    task="invalid",
                )

    def test_prune_network_threshold(self):
        pruned = prune_network(self.adj_df, weight_threshold=0.15)
        self.assertEqual(set(pruned.index), set(self.adj_df.index))
        self.assertEqual(set(pruned.columns), set(self.adj_df.columns))
        full = prune_network(self.adj_df, weight_threshold=0.0)
        self.assertEqual(set(full.index), set(self.adj_df.index))

    def test_prune_network_by_quantile(self):
        pruned_q = prune_network_by_quantile(self.adj_df, quantile=0.5)
        self.assertEqual(set(pruned_q.index), set(self.adj_df.index))
        empty_adj = pd.DataFrame(
            np.zeros((3, 3)),
            index=["x", "y", "z"],
            columns=["x", "y", "z"],
        )
        pruned_empty = prune_network_by_quantile(empty_adj, quantile=0.5)
        self.assertTrue(pruned_empty.equals(empty_adj))

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
        self.assertEqual(set(filtered.index), {"n3"})

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
        self.assertEqual(set(filtered.index), {"c1", "c3"})

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
        self.assertAlmostEqual(df_m_values.loc[0, "B1"], -3.169925, places=5)
        self.assertAlmostEqual(df_m_values.loc[2, "B2"], 0.0, places=5)
        self.assertAlmostEqual(df_m_values.loc[0, "B2"], -19.931567126628412)


if __name__ == "__main__":
    unittest.main()
