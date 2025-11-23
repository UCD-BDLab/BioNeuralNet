import unittest
import pandas as pd
import numpy as np

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

class TestPreprocessFunctions(unittest.TestCase):
    def setUp(self):
        self.clinical = pd.DataFrame({
            "age": [25, np.inf, 45, 60],
            "bmi": [22.0, 28.5, np.nan, 30.0],
            "gender": ["M", "F", None, "F"],
            "smoker": [True, False, True, False],
            "ignore": [1, 1, 1, 1]
        }, index=["s1", "s2", "s3", "s4"])

        self.y_clin = pd.Series([0, 1, 0, 1], index=["s1", "s2", "s3", "s4"])

        self.X_small = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [1.0, 1.0, 1.0, 1.0],
            "f3": [4.0, 3.0, 2.0, 1.0],
            "f4": [10.0, 20.0, 30.0, 40.0],
        }, index=["i1", "i2", "i3", "i4"])
        
        self.df_var = pd.DataFrame({
            "A": [1.0, 3.0, 5.0],
            "B": [2.0, 4.0, 6.0],
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

        self.y_reg = pd.Series([0.1, 0.4, 0.5, 0.8], index=self.X_small.index)

        adj = np.array([
            [1.0, 0.2, 0.0, 0.8],
            [0.2, 1.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, 0.05],
            [0.8, 0.0, 0.05, 1.0],
        ], dtype=float)

        self.adj_df = pd.DataFrame(adj, index=["a", "b", "c", "d"], columns=["a", "b", "c", "d"])

    def test_clean_inf_nan_replaces_and_drops(self):
        df = pd.DataFrame({"x": [1.0, np.inf, 3.0, -np.inf], "y": [np.nan, 2.0, np.nan, 4.0], "z": [5.0, 5.0, 5.0, 5.0]})
        cleaned = clean_inf_nan(df)

        self.assertTrue(np.allclose(cleaned["x"].values, [1.0, 2.0, 3.0, 2.0]))
        self.assertTrue(np.allclose(cleaned["y"].values, [3.0, 2.0, 3.0, 4.0]))
        self.assertNotIn("z", cleaned.columns)
        self.assertFalse(cleaned.isin([np.inf, -np.inf]).any().any())
        self.assertFalse(cleaned.isna().any().any())

    def test_preprocess_clinical_basic(self):
        result = preprocess_clinical(X=self.clinical,y=self.y_clin,top_k=2,scale=False,ignore_columns=["ignore"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 2)
        self.assertNotIn("ignore", result.columns)

    def test_preprocess_clinical_errors(self):
        bad_y = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with self.assertRaises(ValueError):
            preprocess_clinical(self.clinical.iloc[:2, :], bad_y, top_k=1)
        with self.assertRaises(KeyError):
            preprocess_clinical(self.clinical, self.y_clin, ignore_columns=["nonexistent"])

    def test_select_top_k_variance(self):
        top2 = select_top_k_variance(self.X_small, k=2)
        self.assertEqual(set(top2.columns), {"f4", "f1", "f3"}.intersection(set(top2.columns)))
        self.assertEqual(top2.shape[1], 2)
        self.assertTrue((top2.var(axis=0) > 0).all())

    def test_select_top_k_correlation_unsupervised(self):
        unsup = select_top_k_correlation(self.X_small, y=None, top_k=2)
        self.assertEqual(unsup.shape[1], 2)
        self.assertNotIn("f2", unsup.columns)

    def test_select_top_k_correlation_supervised(self):
        sup = select_top_k_correlation(self.X_small, y=self.y_reg, top_k=2)
        self.assertEqual(sup.shape[1], 2)
        bad_y = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with self.assertRaises(ValueError):
            select_top_k_correlation(self.X_small.iloc[:2, :], bad_y, top_k=1)

    def test_select_top_randomforest(self):
        X_num = pd.DataFrame({
            "a": [1, 2, 1, 2],
            "b": [3, 3, 4, 4],
            "c": [0, 1, 0, 1]
        }, index=["p1", "p2", "p3", "p4"])
        y_bin = pd.Series([0, 1, 0, 1], index=X_num.index)
        top_rf = select_top_randomforest(X_num, y_bin, top_k=2, seed=0)
        self.assertEqual(top_rf.shape[1], 2)

        X_mixed = X_num.copy()
        X_mixed["d"] = ["x", "y", "z", "w"]
        with self.assertRaises(ValueError):
            select_top_randomforest(X_mixed, y_bin, top_k=1)

    def test_top_anova_f_features_classification(self):
        X = pd.DataFrame({
            "f1": [0, 1, 0, 1],
            "f2": [5, 5, 5, 5],
            "f3": [2, 3, 2, 3]
        }, index=["s1", "s2", "s3", "s4"])
        y_bin = pd.Series([0, 1, 0, 1], index=X.index)
        top_feats = top_anova_f_features(X, y_bin, max_features=2, alpha=0.05, task="classification")
        self.assertEqual(top_feats.shape[1], 2)
        self.assertIn("f1", top_feats.columns)

        X_reg = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
            "f3": [0.1, 0.2, 0.3, 0.4]
        }, index=["a", "b", "c", "d"])
        y_cont = pd.Series([1.0, 2.0, 3.0, 4.0], index=X_reg.index)
        top_feats_reg = top_anova_f_features(X_reg, y_cont, max_features=1, alpha=0.05, task="regression")
        self.assertEqual(top_feats_reg.shape[1], 1)

        with self.assertRaises(ValueError):
            top_anova_f_features(X_reg, y_cont, max_features=1, alpha=0.05, task="invalid")

    def test_prune_network_threshold(self):
        pruned = prune_network(self.adj_df, weight_threshold=0.15)
        self.assertEqual(set(pruned.index), set(self.adj_df.index))
        self.assertEqual(set(pruned.columns), set(self.adj_df.columns))
        full = prune_network(self.adj_df, weight_threshold=0.0)
        self.assertEqual(set(full.index), set(self.adj_df.index))

    def test_prune_network_by_quantile(self):
        pruned_q = prune_network_by_quantile(self.adj_df, quantile=0.5)
        self.assertEqual(set(pruned_q.index), set(self.adj_df.index))
        empty_adj = pd.DataFrame(np.zeros((3, 3)), index=["x", "y", "z"], columns=["x", "y", "z"])
        pruned_empty = prune_network_by_quantile(empty_adj, quantile=0.5)
        self.assertTrue(pruned_empty.equals(empty_adj))

    def test_network_remove_low_variance(self):
        net = pd.DataFrame({
            "n1": [1.0, 2.0, 1.0],
            "n2": [1.0, 1.0, 1.0],
            "n3": [3.0, 4.0, 5.0]
        }, index=["n1", "n2", "n3"])
        filtered = network_remove_low_variance(net, threshold=0.5)
        self.assertNotIn("n2", filtered.index)
        self.assertNotIn("n2", filtered.columns)
        self.assertEqual(set(filtered.index), {"n3"})

    def test_network_remove_high_zero_fraction(self):
        net = pd.DataFrame({
            "c1": [1.0, 0.5, 0.0],
            "c2": [1.0, 0.0, 0.0],
            "c3": [0.2, 0.2, 0.2]
        }, index=["c1", "c2", "c3"])
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
        self.assertAlmostEqual(df_m_values.loc[0, "B1"], -3.169925)
        self.assertAlmostEqual(df_m_values.loc[2, "B2"], 0.0)
        self.assertAlmostEqual(df_m_values.loc[0, "B2"], -19.931567126628412)

if __name__ == "__main__":
    unittest.main()