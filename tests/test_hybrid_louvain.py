import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd

from bioneuralnet.clustering.hybrid_louvain import HybridLouvain
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.network.generate import correlation_network

class TestHybridLouvain(unittest.TestCase):
    def setUp(self):
        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]
        Y_df = example["Y"]

        self.Y = Y_df.iloc[:, 0] if isinstance(Y_df, pd.DataFrame) else Y_df

        self.B = pd.concat([X1.iloc[:, :15], X2.iloc[:, :15]], axis=1)

        adj = correlation_network(
            self.B, k=3, method="pearson", signed=True,
            normalize=True, mutual=False, per_node=True,
            threshold=None, self_loops=False,
        )
        self.G = nx.from_pandas_adjacency(adj)
        self.node_list = list(self.G.nodes())

    def _make_fake_louvain(self):
        fake = MagicMock()
        partition = {n: 0 for n in self.node_list}
        fake.run.return_value = partition
        fake.get_top_communities.return_value = [(0, 0.7, self.node_list)]
        fake.get_combined_quality.return_value = 0.5
        return fake

    def _make_fake_pagerank(self):
        fake = MagicMock()
        fake.run.return_value = {
            "cluster_nodes": self.node_list,
            "conductance": 0.5,
            "composite_score": 0.6,
        }
        fake.phen_omics_corr.return_value = (0.8,)
        return fake

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_returns_dict_with_expected_keys(self, mock_pr_cls, mock_lou_cls):
        mock_lou_cls.return_value = self._make_fake_louvain()
        mock_pr_cls.return_value = self._make_fake_pagerank()

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        result = hybrid.run(as_dfs=False)

        self.assertIsInstance(result, dict)
        for key in ("best_nodes", "best_correlation", "best_iteration", "iterations", "all_subgraphs"):
            self.assertIn(key, result)

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_best_nodes_and_correlation(self, mock_pr_cls, mock_lou_cls):
        mock_lou_cls.return_value = self._make_fake_louvain()
        mock_pr_cls.return_value = self._make_fake_pagerank()

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        result = hybrid.run(as_dfs=False)

        self.assertIsInstance(result["best_nodes"], list)
        self.assertIsInstance(result["best_correlation"], float)
        self.assertGreaterEqual(result["best_correlation"], 0.0)
        self.assertIsInstance(result["iterations"], list)
        self.assertGreater(len(result["iterations"]), 0)

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_as_dfs_returns_adjacency_dataframes(self, mock_pr_cls, mock_lou_cls):
        mock_lou_cls.return_value = self._make_fake_louvain()
        mock_pr_cls.return_value = self._make_fake_pagerank()

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        result = hybrid.run(as_dfs=True)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for df in result:
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(df.shape[0], df.shape[1])

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_accepts_dataframe_phenotype(self, mock_pr_cls, mock_lou_cls):
        mock_lou_cls.return_value = self._make_fake_louvain()
        mock_pr_cls.return_value = self._make_fake_pagerank()

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y.to_frame())
        result = hybrid.run(as_dfs=False)
        self.assertIsInstance(result, dict)

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_calls_louvain_and_pagerank(self, mock_pr_cls, mock_lou_cls):
        mock_lou_cls.return_value = self._make_fake_louvain()
        mock_pr_cls.return_value = self._make_fake_pagerank()

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        hybrid.run(as_dfs=False)

        mock_lou_cls.assert_called()
        mock_pr_cls.assert_called()

    def test_init_accepts_adjacency_dataframe(self):
        adj_df = nx.to_pandas_adjacency(self.G)
        hybrid = HybridLouvain(G=adj_df, B=self.B, Y=self.Y)
        self.assertIsInstance(hybrid.G_original, nx.Graph)

    def test_init_invalid_graph_raises(self):
        with self.assertRaises(TypeError):
            HybridLouvain(G="not_a_graph", B=self.B, Y=self.Y)

    def test_best_subgraph_before_run_raises(self):
        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        with self.assertRaises(ValueError):
            _ = hybrid.best_subgraph

if __name__ == "__main__":
    unittest.main()
