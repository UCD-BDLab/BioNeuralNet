import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd

from bioneuralnet.clustering.hybrid_louvain import HybridLouvain
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.utils.graph import gen_correlation_graph


class TestHybridLouvain(unittest.TestCase):
    def setUp(self):
        # Using our synthetic example dataset for testing
        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]
        Y_df = example["Y"]

        if isinstance(Y_df, pd.DataFrame):
            self.Y = Y_df.iloc[:, 0]
        else:
            self.Y = Y_df

        # A small subset of features to keep tests light
        X1_small = X1.iloc[:, 0:15]
        X2_small = X2.iloc[:, 0:15]

        self.B = pd.concat([X1_small, X2_small], axis=1)

        # Build a feature–feature graph from the omics data
        adj = gen_correlation_graph(
            self.B,
            k=3,
            method="pearson",
            signed=True,
            normalize=True,
            mutual=False,
            per_node=True,
            threshold=None,
            self_loops=False,
        )

        self.G = nx.from_pandas_adjacency(adj)

        # we store node list so we can build fake partitions over real node names
        self.node_list = []
        for n in self.G.nodes():
            self.node_list.append(n)

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_returns_partition_and_clusters_dict(self, mock_page_rank_cls, mock_louvain_cls):
        fake_louvain = MagicMock()

        # partition: put every node into community 0
        partition = {}
        for n in self.node_list:
            partition[n] = 0
        fake_louvain.run.return_value = partition
        fake_louvain.get_quality.return_value = 0.5

        def fake_compute_corr(nodes):
            return (0.7, None)

        fake_louvain._compute_community_correlation.side_effect = fake_compute_corr
        mock_louvain_cls.return_value = fake_louvain

        fake_pagerank = MagicMock()

        def fake_pr_run(best_seed):
            return {"cluster_nodes": best_seed,"conductance": 0.5,"correlation": 0.8,"composite_score": 0.6}

        fake_pagerank.run.side_effect = fake_pr_run
        mock_page_rank_cls.return_value = fake_pagerank

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y.to_frame())
        result = hybrid.run(as_dfs=False)

        self.assertIsInstance(result, dict)
        self.assertIn("curr", result)
        self.assertIn("clus", result)

        # partition should be exactly what CorrelatedLouvain.run returned
        self.assertEqual(result["curr"], partition)

        # 1 cluster recorded for iteration 0
        clus = result["clus"]
        self.assertIsInstance(clus, dict)
        self.assertIn(0, clus)

        cluster_nodes = clus[0]
        self.assertEqual(len(cluster_nodes), len(self.node_list))

        # Checking that all graph nodes are in the cluster
        for n in self.node_list:
            self.assertIn(n, cluster_nodes)

        mock_louvain_cls.assert_called()
        mock_page_rank_cls.assert_called()

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain", autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank", autospec=True)
    def test_run_as_dfs_returns_list_of_dataframes(self, mock_page_rank_cls, mock_louvain_cls):
        fake_louvain = MagicMock()

        partition = {}
        for n in self.node_list:
            partition[n] = 0
        fake_louvain.run.return_value = partition
        fake_louvain.get_quality.return_value = 0.5
        fake_louvain._compute_community_correlation.side_effect = lambda nodes: (0.7, None)
        mock_louvain_cls.return_value = fake_louvain

        fake_pagerank = MagicMock()
        
        fake_pagerank.run.side_effect = lambda best_seed: {"cluster_nodes": best_seed,"conductance": 0.1,"correlation": 0.1,"composite_score": 0.1}
        mock_page_rank_cls.return_value = fake_pagerank

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        dfs_list = hybrid.run(as_dfs=True)

        self.assertIsInstance(dfs_list, list)
        # we expect a single omics subnetwork DataFrame
        self.assertEqual(len(dfs_list), 1)

        df0 = dfs_list[0]
        self.assertIsInstance(df0, pd.DataFrame)

        # Columns should match the omics columns used to build the graph
        cols_set = set(df0.columns)
        expected_cols_set = set(self.B.columns)
        self.assertEqual(cols_set, expected_cols_set)

        # And values should match the original subset of omics data
        pd.testing.assert_frame_equal(df0, self.B.loc[:, df0.columns])

        mock_louvain_cls.assert_called()
        mock_page_rank_cls.assert_called()

if __name__ == "__main__":
    unittest.main()
