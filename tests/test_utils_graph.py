import unittest
import pandas as pd
import numpy as np

from bioneuralnet.utils.graph import gen_similarity_graph
from bioneuralnet.utils.graph import gen_correlation_graph
from bioneuralnet.utils.graph import gen_threshold_graph
from bioneuralnet.utils.graph import gen_gaussian_knn_graph
from bioneuralnet.utils.graph import gen_lasso_graph
from bioneuralnet.utils.graph import gen_mst_graph
from bioneuralnet.utils.graph import gen_snn_graph

class TestUtilsGraph(unittest.TestCase):
    def setUp(self):
        np.random.seed(1998)
        self.X = pd.DataFrame(np.random.rand(10, 5), columns=["n1", "n2", "n3", "n4", "n5"])
        self.N = self.X.shape[1]

    def _basic_checks(self, G: pd.DataFrame):
        self.assertIsInstance(G, pd.DataFrame)
        self.assertEqual(G.shape, (self.N, self.N))
        self.assertEqual(list(G.index), list(self.X.columns))
        self.assertEqual(list(G.columns), list(self.X.columns))
        self.assertTrue(np.all(G.values >= 0))
        
        row_sums = G.sum(axis=1).values
        is_normalized = np.isclose(row_sums, 1.0, atol=1e-5) | np.isclose(row_sums, 0.0, atol=1e-5)
        self.assertTrue(is_normalized.all())

    def test_gen_similarity_graph_default(self):
        G = gen_similarity_graph(self.X, k=2)
        self._basic_checks(G)

    def test_gen_similarity_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_similarity_graph(np.array([[1, 2], [3, 4]]))

    def test_gen_correlation_graph_default(self):
        G = gen_correlation_graph(self.X, k=2, method="pearson")
        self._basic_checks(G)

    def test_gen_correlation_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_correlation_graph("not a df")

    def test_gen_threshold_graph_default(self):
        G = gen_threshold_graph(self.X, b=2.0, k=2)
        self._basic_checks(G)

    def test_gen_threshold_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_threshold_graph(None)

    def test_gen_gaussian_knn_graph_default(self):
        G = gen_gaussian_knn_graph(self.X, k=2)
        self._basic_checks(G)

    def test_gen_gaussian_knn_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_gaussian_knn_graph(123)

    def test_gen_lasso_graph_default(self):
        G = gen_lasso_graph(self.X, alpha=0.1)
        self._basic_checks(G)

    def test_gen_lasso_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_lasso_graph([1, 2, 3])

    def test_gen_mst_graph_default(self):
        G = gen_mst_graph(self.X)
        self._basic_checks(G)

    def test_gen_mst_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_mst_graph(5.0)

    def test_gen_snn_graph_default(self):
        G = gen_snn_graph(self.X, k=2)
        self._basic_checks(G)

    def test_gen_snn_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_snn_graph("oops, not a df")

if __name__ == "__main__":
    unittest.main()