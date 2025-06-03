import unittest
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from bioneuralnet.metrics import omics_correlation
from bioneuralnet.metrics import cluster_correlation
from bioneuralnet.metrics import louvain_to_adjacency

class TestDataFunctions(unittest.TestCase):
    def setUp(self):
        self.omics = pd.DataFrame({
            "g1": [1.0, 2.0, 3.0, 4.0],
            "g2": [2.0, 4.0, 6.0, 8.0]
        }, index=["s1", "s2", "s3", "s4"])
        self.pheno = pd.DataFrame({"phen": [10.0, 20.0, 30.0, 40.0]}, index=["s1", "s2", "s3", "s4"])

    def test_omics_correlation_valid(self):
        corr, pval = omics_correlation(self.omics, self.pheno)
        expected_pc1 = np.array([ -1.34164, -0.44721, 0.44721, 1.34164 ])
        expected_corr, _ = pearsonr(expected_pc1, self.pheno["phen"].values)
        self.assertAlmostEqual(corr, expected_corr, places=5)
        self.assertTrue(0 <= pval <= 1)

    def test_omics_correlation_empty(self):
        empty = pd.DataFrame()
        with self.assertRaises(ValueError):
            omics_correlation(empty, self.pheno)
        with self.assertRaises(ValueError):
            omics_correlation(self.omics, pd.DataFrame())

    def test_omics_correlation_mismatch(self):
        pheno_short = pd.DataFrame({"phen": [1.0, 2.0]}, index=["s1", "s2"])
        with self.assertRaises(ValueError):
            omics_correlation(self.omics, pheno_short)

    def test_cluster_correlation_size_one(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=["x", "y", "z"])
        size, corr = cluster_correlation(df, pd.DataFrame({"p": [1.0, 2.0, 3.0]}, index=["x", "y", "z"]))
        self.assertEqual(size, 1)
        self.assertIsNone(corr)

    def test_cluster_correlation_zero_variance(self):
        df = pd.DataFrame({
            "c1": [1.0, 1.0, 1.0],
            "c2": [2.0, 2.0, 2.0]
        }, index=["i1", "i2", "i3"])
        size, corr = cluster_correlation(df, pd.DataFrame({"p": [1.0, 2.0, 3.0]}, index=["i1", "i2", "i3"]))
        self.assertEqual(size, 2)
        self.assertIsNone(corr)

    def test_cluster_correlation_insufficient_samples(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0],
            "f2": [2.0, 4.0]
        }, index=["s1", "s2"])
        ph = pd.DataFrame({"p": [5.0, 10.0]}, index=["s1", "s2"])
        size, corr = cluster_correlation(df, ph)
        self.assertEqual(size, 2)
        self.assertIsNone(corr)

    def test_cluster_correlation_valid(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 4.0, 6.0, 8.0]
        }, index=["s1", "s2", "s3", "s4"])
        ph = pd.DataFrame({"p": [5.0, 10.0, 15.0, 20.0]}, index=["s1", "s2", "s3", "s4"])
        size, corr = cluster_correlation(df, ph)
        self.assertEqual(size, 2)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_louvain_to_adjacency(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 4.0, 6.0],
            "c": [1.0, 0.0, 1.0]
        }, index=["x", "y", "z"])
        adj = louvain_to_adjacency(df)
        self.assertEqual(set(adj.columns), {"a", "b", "c"})
        self.assertEqual(set(adj.index), {"a", "b", "c"})
        self.assertTrue((np.diag(adj.values) == 0).all())
        self.assertFalse(adj.isna().any().any())

if __name__ == "__main__":
    unittest.main()
