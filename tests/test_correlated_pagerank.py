import unittest
import networkx as nx
import pandas as pd
import numpy as np
from bioneuralnet.clustering import CorrelatedPageRank

class TestCorrelatedPageRank(unittest.TestCase):
    def setUp(self):
        self.G = nx.complete_graph(4, create_using=nx.DiGraph())
        nodes = list(self.G.nodes())

        data = {node: np.random.rand(10) for node in nodes}
        self.B = pd.DataFrame(data)

        self.Y = pd.DataFrame({"phenotype": np.random.rand(10)})

    def test_run_valid(self):
        cp = CorrelatedPageRank(
            self.G, self.B, self.Y,
            alpha=0.9, max_iter=100, tol=1e-6, k=0.5, tune=False
        )

        seed_nodes = [list(self.G.nodes())[0], list(self.G.nodes())[1]]
        res = cp.run(seed_nodes)
        expected_keys = [
            "cluster_nodes", "cluster_size", "conductance",
            "correlation", "composite_score", "correlation_pvalue"
        ]
        for key in expected_keys:
            self.assertIn(key, res)

    def test_run_empty_seed(self):
        cp = CorrelatedPageRank(
            self.G, self.B, self.Y,
            alpha=0.9, max_iter=100, tol=1e-6, k=0.5, tune=False
        )
        with self.assertRaises(ValueError):
            cp.run([])

if __name__ == "__main__":
    unittest.main()
