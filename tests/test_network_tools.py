import unittest
import pandas as pd
import numpy as np

from bioneuralnet.network.tools import NetworkAnalyzer
from bioneuralnet.network.tools import network_search
from bioneuralnet.datasets import DatasetLoader

class TestNetworkAnalyzer(unittest.TestCase):
    def setUp(self):
        self.adj = pd.DataFrame(
            [
                [0.0, 0.8, 0.2, 0.0],
                [0.8, 0.0, 0.5, 0.1],
                [0.2, 0.5, 0.0, 0.9],
                [0.0, 0.1, 0.9, 0.0],
            ],
            index=["n1", "n2", "n3", "n4"],
            columns=["n1", "n2", "n3", "n4"],
        )
        self.analyzer = NetworkAnalyzer(self.adj)

    def test_basic_statistics_returns_dict(self):
        stats = self.analyzer.basic_statistics(threshold=0.5)
        self.assertIsInstance(stats, dict)
        for key in ("nodes", "edges", "density", "avg_degree", "isolated"):
            self.assertIn(key, stats)
        self.assertEqual(stats["nodes"], 4)

    def test_degree_distribution_returns_dataframe(self):
        df = self.analyzer.degree_distribution(threshold=0.5)
        self.assertIsInstance(df, pd.DataFrame)
        for col in ("degree", "count", "percentage"):
            self.assertIn(col, df.columns)
        self.assertAlmostEqual(df["percentage"].sum(), 100.0, places=5)

    def test_hub_analysis_returns_top_n(self):
        top_n = 3
        df = self.analyzer.hub_analysis(threshold=0.1, top_n=top_n)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertLessEqual(len(df), top_n)
        self.assertIn("degree", df.columns)
        self.assertIn("feature", df.columns)

    def test_edge_weight_analysis_returns_array(self):
        weights = self.analyzer.edge_weight_analysis()
        self.assertIsInstance(weights, np.ndarray)
        self.assertGreater(len(weights), 0)
        self.assertTrue((weights > 0).all())

    def test_find_strongest_edges_returns_dataframe(self):
        top_n = 3
        df = self.analyzer.find_strongest_edges(top_n=top_n)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertLessEqual(len(df), top_n)
        for col in ("feature1", "feature2", "weight"):
            self.assertIn(col, df.columns)
        # weights should be sorted descending
        self.assertTrue((df["weight"].diff().dropna() <= 0).all())

    def test_connected_components_returns_dict(self):
        result = self.analyzer.connected_components(threshold=0.1)
        self.assertIsInstance(result, dict)
        self.assertIn("n_components", result)
        self.assertIn("labels", result)
        self.assertIn("sizes", result)
        self.assertEqual(len(result["labels"]), 4)

    def test_source_omics_assignment(self):
        omics1 = pd.DataFrame(np.random.rand(5, 2), columns=["n1", "n2"])
        omics2 = pd.DataFrame(np.random.rand(5, 2), columns=["n3", "n4"])
        analyzer = NetworkAnalyzer(self.adj, source_omics=[omics1, omics2])
        self.assertIn("omic_1", analyzer.omics_types)
        self.assertIn("omic_2", analyzer.omics_types)

class TestNetworkSearch(unittest.TestCase):
    def setUp(self):
        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]
        y_ex = example["Y"]

        if isinstance(y_ex, pd.DataFrame):
            y_ex = y_ex.iloc[:, 0]

        y_binary = (y_ex > float(np.median(y_ex.values))).astype(int)

        self.omics_data = pd.concat([X1, X2], axis=1)
        self.y = y_binary

    def test_network_search_returns_tuple(self):
        best_G, best_params, results = network_search(
            self.omics_data,
            self.y,
            methods=["correlation", "threshold"],
            trials=2,
            verbose=False,
        )
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(best_G, pd.DataFrame)
        self.assertIsInstance(best_params, dict)

    def test_network_search_results_columns(self):
        _, _, results = network_search(
            self.omics_data,
            self.y,
            methods=["correlation"],
            trials=2,
            verbose=False,
        )
        for col in ("method", "score", "f1", "topology"):
            self.assertIn(col, results.columns)

    def test_network_search_graph_shape(self):
        best_G, _, _ = network_search(
            self.omics_data,
            self.y,
            methods=["threshold"],
            trials=2,
            verbose=False,
        )
        n_features = self.omics_data.shape[1]
        self.assertEqual(best_G.shape, (n_features, n_features))


if __name__ == "__main__":
    unittest.main()
