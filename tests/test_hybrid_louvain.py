import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd
from bioneuralnet.clustering.hybrid_louvain import HybridLouvain

class TestHybridLouvain(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(["a", "b", "c"])
        self.B = pd.DataFrame({"a": [1.0, 2.0],"b": [3.0, 4.0],"c": [5.0, 6.0],},index=["sample1", "sample2"],)
        self.Y = pd.Series({"a": 0.0, "b": 1.0, "c": 2.0},name="phenotype")

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain",autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank",autospec=True)
    def test_run_returns_partition_and_clusters_dict(self,mock_page_rank_cls,mock_louvain_cls):
        fake_louvain = MagicMock()
        fake_louvain.run.return_value = {"a": 0, "b": 0, "c": 0}
        fake_louvain.get_quality.return_value = 0.5

        def fake_compute_corr(nodes):
            return (0.7, None)
        
        fake_louvain._compute_community_correlation.side_effect = fake_compute_corr
        mock_louvain_cls.return_value = fake_louvain

        fake_pagerank = MagicMock()
        def fake_pr_run(best_seed):
            return {"cluster_nodes": best_seed}
        
        fake_pagerank.run.side_effect = fake_pr_run
        mock_page_rank_cls.return_value = fake_pagerank

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y.to_frame())
        result = hybrid.run(as_dfs=False)

        expected_partition = {"a": 0, "b": 0, "c": 0}
        self.assertIn("curr", result)
        self.assertEqual(result["curr"], expected_partition)

        self.assertIn("clus", result)
        self.assertEqual(set(result["clus"].keys()), {0})
        self.assertEqual(result["clus"][0], ["a", "b", "c"])

        mock_louvain_cls.assert_called()
        mock_page_rank_cls.assert_called()

    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedLouvain",autospec=True)
    @patch("bioneuralnet.clustering.hybrid_louvain.CorrelatedPageRank",autospec=True)
    def test_run_as_dfs_returns_list_of_dataframes(self,mock_page_rank_cls,mock_louvain_cls):
        fake_louvain = MagicMock()
        fake_louvain.run.return_value = {"a": 0, "b": 0, "c": 0}
        fake_louvain.get_quality.return_value = 0.5
        fake_louvain._compute_community_correlation.side_effect = lambda nodes: (0.7, None)
        mock_louvain_cls.return_value = fake_louvain

        fake_pagerank = MagicMock()
        fake_pagerank.run.side_effect = lambda best_seed: {"cluster_nodes": best_seed}
        mock_page_rank_cls.return_value = fake_pagerank

        hybrid = HybridLouvain(G=self.G, B=self.B, Y=self.Y)
        dfs_list = hybrid.run(as_dfs=True)

        self.assertIsInstance(dfs_list, list)
        self.assertEqual(len(dfs_list), 1)

        df0 = dfs_list[0]
        self.assertIsInstance(df0, pd.DataFrame)
        self.assertEqual(set(df0.columns), {"a", "b", "c"})
        pd.testing.assert_frame_equal(df0, self.B.loc[:, ["a", "b", "c"]])

if __name__ == "__main__":
    unittest.main()
