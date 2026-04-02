import unittest
import pandas as pd
import numpy as np

from bioneuralnet.metrics.correlation import (
    omics_correlation,
    cluster_pca_correlation,
    cluster_correlation,
)

class TestCorrelationMetrics(unittest.TestCase):
    def setUp(self):
        self.omics = pd.DataFrame({
            "g1": [1.0, 2.0, 3.0, 4.0],
            "g2": [2.0, 4.0, 6.0, 8.0],
        }, index=["s1", "s2", "s3", "s4"])
        self.pheno = pd.DataFrame(
            {"phen": [10.0, 20.0, 30.0, 40.0]},
            index=["s1", "s2", "s3", "s4"],
        )

    def test_omics_correlation_valid(self):
        corr, pval = omics_correlation(self.omics, self.pheno)
        self.assertIsInstance(corr, float)
        self.assertTrue(0 <= pval <= 1)
        self.assertAlmostEqual(abs(corr), 1.0, places=5)

    def test_omics_correlation_empty_raises(self):
        with self.assertRaises(ValueError):
            omics_correlation(pd.DataFrame(), self.pheno)
        with self.assertRaises(ValueError):
            omics_correlation(self.omics, pd.DataFrame())

    def test_omics_correlation_length_mismatch_raises(self):
        short_pheno = pd.DataFrame({"phen": [1.0, 2.0]}, index=["s1", "s2"])
        with self.assertRaises(ValueError):
            omics_correlation(self.omics, short_pheno)

    def test_cluster_pca_correlation_valid(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 4.0, 6.0, 8.0],
        }, index=["s1", "s2", "s3", "s4"])
        ph = pd.DataFrame({"p": [5.0, 10.0, 15.0, 20.0]}, index=["s1", "s2", "s3", "s4"])
        size, corr = cluster_pca_correlation(df, ph)
        self.assertEqual(size, 2)
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(abs(corr), 1.0, places=5)

    def test_cluster_pca_correlation_size_one(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=["x", "y", "z"])
        ph = pd.DataFrame({"p": [1.0, 2.0, 3.0]}, index=["x", "y", "z"])
        size, corr = cluster_pca_correlation(df, ph)
        self.assertEqual(size, 1)
        self.assertIsNone(corr)

    def test_cluster_pca_correlation_zero_variance(self):
        df = pd.DataFrame({
            "c1": [1.0, 1.0, 1.0],
            "c2": [2.0, 2.0, 2.0],
        }, index=["i1", "i2", "i3"])
        ph = pd.DataFrame({"p": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"])
        size, corr = cluster_pca_correlation(df, ph)
        self.assertEqual(size, 2)
        self.assertIsNone(corr)

    def test_cluster_pca_correlation_insufficient_samples(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0],
            "f2": [2.0, 4.0],
        }, index=["s1", "s2"])
        ph = pd.DataFrame({"p": [5.0, 10.0]}, index=["s1", "s2"])
        size, corr = cluster_pca_correlation(df, ph)
        self.assertEqual(size, 2)
        self.assertIsNone(corr)

    def test_cluster_correlation_returns_adjacency(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 4.0, 6.0],
            "c": [1.0, 0.0, 1.0],
        }, index=["x", "y", "z"])
        adj = cluster_correlation(df)
        self.assertIsInstance(adj, pd.DataFrame)
        self.assertEqual(set(adj.columns), {"a", "b", "c"})
        self.assertEqual(set(adj.index), {"a", "b", "c"})

    def test_cluster_correlation_diagonal_zero(self):
        adj = cluster_correlation(self.omics)
        self.assertTrue((np.diag(adj.values) == 0).all())

    def test_cluster_correlation_no_nans(self):
        adj = cluster_correlation(self.omics)
        self.assertFalse(adj.isna().any().any())

    def test_cluster_correlation_symmetric(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 3.0, 2.0, 4.0],
        }, index=["s1", "s2", "s3", "s4"])
        adj = cluster_correlation(df)
        pd.testing.assert_frame_equal(adj, adj.T)

if __name__ == "__main__":
    unittest.main()
