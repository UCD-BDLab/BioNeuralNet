import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bioneuralnet.metrics import omics_correlation, cluster_correlation, louvain_to_adjacency
from bioneuralnet.metrics import evaluate_rf
from bioneuralnet.metrics import plot_variance_distribution, plot_variance_by_feature, plot_performance, plot_embeddings, plot_network, compare_clusters

class TestCorrelationFunctions(unittest.TestCase):
    def test_omics_correlation_valid(self):
        df_omics = pd.DataFrame(np.random.rand(100, 5), columns=["gene0", "gene1", "gene2", "gene3", "gene4"])
        df_pheno = pd.DataFrame(np.random.rand(100), columns=["phenotype"])
        corr, pval = omics_correlation(df_omics, df_pheno)
        self.assertIsInstance(corr, float)
        self.assertIsInstance(pval, float)

    def test_omics_correlation_empty(self):
        df_omics = pd.DataFrame()
        df_pheno = pd.DataFrame([1, 2, 3], columns=["phenotype"])
        with self.assertRaises(ValueError):
            omics_correlation(df_omics, df_pheno)

    def test_omics_correlation_mismatched_length(self):
        df_omics = pd.DataFrame(np.random.rand(10, 3))
        df_pheno = pd.DataFrame(np.random.rand(5), columns=["phenotype"])
        with self.assertRaises(ValueError):
            omics_correlation(df_omics, df_pheno)

    def test_cluster_correlation_small_cluster(self):
        df_cluster = pd.DataFrame({"A": [1, 2, 3, 4]})
        df_pheno = pd.DataFrame({"phenotype": [1, 2, 3, 4]})
        size, corr = cluster_correlation(df_cluster, df_pheno)
        self.assertEqual(size, 1)
        self.assertIsNone(corr)

    def test_louvain_to_adjacency(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 1], "C": [3, 1, 2]})
        adj = louvain_to_adjacency(df)
        self.assertEqual(adj.shape, df.shape)
        np.testing.assert_array_equal(np.diag(adj.values), np.zeros(adj.shape[0]))

class TestEvaluationFunction(unittest.TestCase):
    def test_evaluate_rf_regression(self):
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=0)
        r2 = evaluate_rf(X, y, n=10, mode="regression")
        self.assertIsInstance(r2, float)
        self.assertTrue(-1.0 <= r2 <= 1.0)

    def test_evaluate_rf_classification(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=0)
        acc = evaluate_rf(X, y, n=10, mode="classification")
        self.assertIsInstance(acc, float)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_evaluate_rf_invalid_mode(self):
        X = [[1]]
        y = [1]
        with self.assertRaises(ValueError):
            evaluate_rf(X, y, mode="invalid")

class TestPlotFunctions(unittest.TestCase):
    def test_plot_variance_distribution(self):
        df = pd.DataFrame({"A": np.random.rand(100), "B": np.random.rand(100)})
        fig = plot_variance_distribution(df, bins=10)
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_variance_by_feature(self):
        df = pd.DataFrame({"A": np.random.rand(100), "B": np.random.rand(100)})
        fig = plot_variance_by_feature(df)
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_performance(self):
        embedding_result = pd.DataFrame({"Actual": [0, 1, 1, 0], "Predicted": [0, 1, 0, 0]})
        raw_rf_acc = 0.75
        orig_show = plt.show
        plt.show = lambda: None
        try:
            plot_performance(embedding_result, raw_rf_acc, title="Test Performance")
        finally:
            plt.show = orig_show

    def test_plot_embeddings(self):
        embeddings = np.random.rand(50, 10)
        orig_show = plt.show
        plt.show = lambda: None
        try:
            plot_embeddings(embeddings)
        finally:
            plt.show = orig_show

    def test_plot_network(self):
        df = pd.DataFrame({"A": [1, 0.5, 0.2], "B": [0.5, 1, 0.3], "C": [0.2, 0.3, 1]}, index=["A", "B", "C"])
        orig_show = plt.show
        plt.show = lambda: None
        try:
            mapping_df = plot_network(df, weight_threshold=0.1, show_labels=False, show_edge_weights=False)
            self.assertIsInstance(mapping_df, pd.DataFrame)
        finally:
            plt.show = orig_show

    def test_compare_clusters(self):
        cluster1 = pd.DataFrame({"gene1": [1, 2, 3], "gene2": [2, 3, 4]}, index=[0, 1, 2])
        cluster2 = pd.DataFrame({"gene3": [1, 2, 3], "gene4": [3, 4, 5]}, index=[0, 1, 2])
        pheno = pd.DataFrame({"phenotype": [1, 2, 3]})
        omics_merged = pd.DataFrame({"gene3": [1, 2, 3], "gene4": [3, 4, 5]}, index=[0, 1, 2])
        orig_show = plt.show
        plt.show = lambda: None
        try:
            df_results = compare_clusters([cluster1], [cluster2], pheno, omics_merged, label1="Test1", label2="Test2")
            expected_cols = ["Cluster", "Louvain Size", "Louvain Correlation", "SMCCNET Size", "SMCCNET Correlation"]
            i = 0
            while i < len(expected_cols):
                self.assertIn(expected_cols[i], df_results.columns)
                i = i + 1
        finally:
            plt.show = orig_show

if __name__ == "__main__":
    unittest.main()
