import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold

from bioneuralnet.utils.graph_tools import graph_analysis
from bioneuralnet.utils.graph_tools import repair_graph_connectivity
from bioneuralnet.utils.graph_tools import find_optimal_graph
from bioneuralnet.utils.graph_tools import _find_optimal_epsilon
from bioneuralnet.utils.graph_tools import _feature_proxy

class TestGraphTools(unittest.TestCase):
    def setUp(self):
        self.adj_disconnected = pd.DataFrame(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0]
            ],
            index=["n1", "n2", "n3", "n4"],
            columns=["n1", "n2", "n3", "n4"]
        )

        self.omics_data = pd.DataFrame(
            np.random.rand(10, 4),
            index=[f"s{i}" for i in range(10)],
            columns=["n1", "n2", "n3", "n4"]
        )
        
        self.y = pd.Series([0]*5 + [1]*5, index=self.omics_data.index)
        self.omics_list = [self.omics_data[["n1", "n2"]], self.omics_data[["n3", "n4"]]]

    @patch('bioneuralnet.utils.graph_tools.logger')
    def test_graph_analysis_logs(self, mock_logger):
        graph_analysis(self.adj_disconnected, "TestGraph", omics_list=None)
        self.assertTrue(mock_logger.info.called)
        
        calls = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Nodes: 4" in c for c in calls))
        self.assertTrue(any("Connected components: 2" in c for c in calls))

    def test_repair_graph_connectivity_eigen(self):
        eps = 0.01
        repaired = repair_graph_connectivity(
            self.adj_disconnected, 
            epsilon=eps, 
            selection_mode="eigen"
        )
        
        self.assertIsInstance(repaired, pd.DataFrame)
        self.assertEqual(repaired.shape, (4, 4))
        
        G = nx.from_pandas_adjacency(repaired)
        self.assertEqual(nx.number_connected_components(G), 1)
        
        original_sum = self.adj_disconnected.values.sum()
        repaired_sum = repaired.values.sum()
        self.assertGreater(repaired_sum, original_sum)

    def test_repair_graph_connectivity_modality_corr(self):
        eps = 0.01
        repaired = repair_graph_connectivity(
            self.adj_disconnected,
            epsilon=eps,
            selection_mode="modality_corr",
            omics_list=self.omics_list
        )
        
        G = nx.from_pandas_adjacency(repaired)
        self.assertEqual(nx.number_connected_components(G), 1)

    def test_find_optimal_epsilon(self):
        eps_list = _find_optimal_epsilon(self.adj_disconnected, n_eps=5)
        self.assertIsInstance(eps_list, list)
        self.assertEqual(len(eps_list), 5)
        self.assertTrue(all(isinstance(x, float) for x in eps_list))
        self.assertEqual(eps_list, sorted(eps_list))

    def test_feature_proxy(self):
        cv = StratifiedKFold(n_splits=2)
        mean_f1, std_f1 = _feature_proxy(
            self.adj_disconnected, 
            self.omics_data.T, 
            self.y, 
            cv, 
            mode="eigenvector"
        )
        self.assertIsInstance(mean_f1, float)
        self.assertIsInstance(std_f1, float)

    def test_find_optimal_graph_execution(self):
        best_G, best_params, results = find_optimal_graph(
            self.omics_data,
            self.y,
            methods=['correlation', 'threshold'],
            trials=2,
            verbose=False
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("method", results.columns)
        self.assertIn("score", results.columns)
        
        if best_G is not None:
            self.assertIsInstance(best_G, pd.DataFrame)
            self.assertIsInstance(best_params, dict)
            self.assertEqual(best_G.shape[0], self.omics_data.shape[1])

if __name__ == "__main__":
    unittest.main()