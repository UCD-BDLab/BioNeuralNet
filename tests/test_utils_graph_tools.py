import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import networkx as nx
import warnings

from bioneuralnet.utils.graph_tools import graph_analysis
from bioneuralnet.utils.graph_tools import repair_graph_connectivity
from bioneuralnet.utils.graph_tools import find_optimal_graph
from bioneuralnet.datasets import DatasetLoader


class TestGraphTools(unittest.TestCase):
    def setUp(self):
        self.adj_disconnected = pd.DataFrame(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            index=["n1", "n2", "n3", "n4"],
            columns=["n1", "n2", "n3", "n4"],
        )

        index_list = []
        i = 0
        while i < 10:
            index_list.append("s" + str(i))
            i += 1

        self.omics_data = pd.DataFrame(
            np.random.rand(10, 4),
            index=index_list,
            columns=["n1", "n2", "n3", "n4"],
        )

        y_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.y = pd.Series(y_values, index=self.omics_data.index)

        self.omics_list = []
        self.omics_list.append(self.omics_data[["n1", "n2"]])
        self.omics_list.append(self.omics_data[["n3", "n4"]])

        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]
        y_ex = example["Y"]

        if isinstance(y_ex, pd.DataFrame):
            y_ex = y_ex.iloc[:, 0]

        y_array = y_ex.values
        median_val = float(np.median(y_array))
        y_binary = (y_ex > median_val).astype(int)

        self.omics_data_example = pd.concat([X1, X2], axis=1)
        self.y_example = y_binary

    @patch("bioneuralnet.utils.graph_tools.logger")
    def test_graph_analysis_logs(self, mock_logger):
        graph_analysis(self.adj_disconnected, "TestGraph", omics_list=None)
        self.assertTrue(mock_logger.info.called)

        calls = []
        for c in mock_logger.info.call_args_list:
            calls.append(str(c))

        found_nodes = False
        found_components = False
        for c in calls:
            if "Nodes: 4" in c:
                found_nodes = True
            if "Connected components: 2" in c:
                found_components = True

        self.assertTrue(found_nodes)
        self.assertTrue(found_components)

    def test_repair_graph_connectivity_eigen(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            eps = 0.01
            repaired = repair_graph_connectivity(
                self.adj_disconnected,
                epsilon=eps,
                selection_mode="eigen",
            )

        self.assertIsInstance(repaired, pd.DataFrame)
        self.assertEqual(repaired.shape, (4, 4))

        G = nx.from_pandas_adjacency(repaired)
        self.assertEqual(nx.number_connected_components(G), 1)

        original_sum = float(self.adj_disconnected.values.sum())
        repaired_sum = float(repaired.values.sum())
        self.assertGreater(repaired_sum, original_sum)

    def test_repair_graph_connectivity_modality_corr(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            eps = 0.01
            repaired = repair_graph_connectivity(
                self.adj_disconnected,
                epsilon=eps,
                selection_mode="modality_corr",
                omics_list=self.omics_list,
            )

        G = nx.from_pandas_adjacency(repaired)
        self.assertEqual(nx.number_connected_components(G), 1)

    def test_find_optimal_graph_execution(self):
        best_G, best_params, results = find_optimal_graph(
            self.omics_data_example,
            self.y_example,
            methods=["correlation", "threshold"],
            trials=2,
            verbose=False,
        )

        self.assertIsInstance(results, pd.DataFrame)

        if not results.empty:
            has_method = "method" in results.columns
            has_score = "score" in results.columns
            self.assertTrue(has_method)
            self.assertTrue(has_score)

        if best_G is not None:
            self.assertIsInstance(best_G, pd.DataFrame)
            self.assertIsInstance(best_params, dict)
            self.assertEqual(best_G.shape[0], self.omics_data_example.shape[1])


if __name__ == "__main__":
    unittest.main()
