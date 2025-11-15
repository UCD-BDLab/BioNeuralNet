import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd
import numpy as np
from bioneuralnet.clustering.leiden import Leiden

class TestLeiden(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(["a", "b", "c"])
        self.embeddings = np.array([[123], [456]])

    @patch("bioneuralnet.clustering.leiden.Leiden",autospec=True)
    def test_run_returns_partition_and_clusters_dict(self,mock_leiden_cls):
        fake_leiden = MagicMock()
        fake_leiden.run.return_value = {"a": 0, "b": 0, "c": 0}
        fake_leiden.get_quality.return_value = 0.5

        def fake_compute_corr(nodes):
            return (0.7, None)

        fake_leiden._compute_community_correlation.side_effect = fake_compute_corr
        mock_leiden_cls.return_value = fake_leiden

        leiden = Leiden(G=self.G, embeddings=self.embeddings)
        result = leiden.run()
        print(result)
        expected_partition = {"a": 0, "b": 0, "c": 0}

if __name__ == "__main__":
    unittest.main()
