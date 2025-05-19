import unittest
import networkx as nx
import pandas as pd
import numpy as np
from bioneuralnet.clustering import CorrelatedLouvain

class TestCorrelatedLouvain(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        nodes = []
        nodes.append("node1")
        nodes.append("node2")
        nodes.append("node3")
        nodes.append("node4")
        nodes.append("node5")
        i = 0
        while i < len(nodes):
            self.G.add_node(nodes[i])
            i = i + 1
        self.G.add_edge("node1", "node2", weight=1.0)
        self.G.add_edge("node2", "node3", weight=1.0)
        self.G.add_edge("node3", "node4", weight=1.0)
        self.G.add_edge("node4", "node5", weight=1.0)
        self.G.add_edge("node5", "node1", weight=1.0)
        data = {}
        i = 0
        while i < len(nodes):
            col = nodes[i]
            vals = []
            j = 0
            while j < 10:
                vals.append(np.random.rand())
                j = j + 1
            data[col] = vals
            i = i + 1
        self.B = pd.DataFrame(data)
        phenos = []
        i = 0
        while i < 10:
            phenos.append(np.random.rand())
            i = i + 1
        self.Y = pd.DataFrame({"phenotype": phenos})

    def test_run_partition(self):
        cl = CorrelatedLouvain(self.G, self.B, self.Y, k3=0.2, k4=0.8, weight="weight", tune=False)
        part = cl.run(as_dfs=False)
        self.assertIsInstance(part, dict)
        keys = []
        for k in part:
            keys.append(k)
        self.assertTrue(len(keys) > 0)

    def test_run_dfs(self):
        cl = CorrelatedLouvain(self.G, self.B, self.Y, k3=0.2, k4=0.8, weight="weight", tune=False)
        dfs = cl.run(as_dfs=True)
        self.assertIsInstance(dfs, list)
        i = 0
        while i < len(dfs):
            self.assertIsInstance(dfs[i], pd.DataFrame)
            i = i + 1

    def test_get_quality(self):
        cl = CorrelatedLouvain(self.G, self.B, self.Y, k3=0.2, k4=0.8, weight="weight", tune=False)
        cl.run(as_dfs=False)
        q = cl.get_quality()
        self.assertIsInstance(q, float)

if __name__ == "__main__":
    unittest.main()
