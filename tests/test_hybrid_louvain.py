import unittest
import networkx as nx
import pandas as pd
import numpy as np
from bioneuralnet.clustering import HybridLouvain

class TestHybridLouvain(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        nodes = []
        nodes.append("a")
        nodes.append("b")
        nodes.append("c")
        nodes.append("d")
        nodes.append("e")
        i = 0
        while i < len(nodes):
            self.G.add_node(nodes[i])
            i = i + 1
        self.G.add_edge("a", "b", weight=1.0)
        self.G.add_edge("b", "c", weight=1.0)
        self.G.add_edge("c", "d", weight=1.0)
        self.G.add_edge("d", "e", weight=1.0)
        self.G.add_edge("e", "a", weight=1.0)
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
        
    def test_run(self):
        hl = HybridLouvain(self.G, self.B, self.Y, k3=0.2, k4=0.8, max_iter=5, weight="weight", tune=False)
        res = hl.run()
        keys = []
        for key in res:
            keys.append(key)
        self.assertIn("curr", keys)
        self.assertIn("clus", keys)
        self.assertIsInstance(res["clus"], dict)

if __name__ == "__main__":
    unittest.main()
