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

class TestGraphGeneration(unittest.TestCase):
    def setUp(self):
        # for testing we use a small dataframe, 3 rows (nodes) and 2 columns (features) each.
        # distinct values so that results are deterministic
        self.X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}, index=["n1", "n2", "n3"])
        self.N = 3

    def _basic_checks(self, G: pd.DataFrame):
        """
        Shared checks for any generated graph:

            - Must be a DataFrame of shape (N, N)
            - Rows sum to 1 (with small tolerance)
            - Diagonal entries (self-loops) are > 0

        """

        self.assertIsInstance(G, pd.DataFrame)
        self.assertEqual(G.shape, (self.N, self.N))
        row_sums = G.sum(axis=1).values
        self.assertTrue(np.allclose(row_sums, np.ones(self.N), atol=1e-6))
        diag = np.diag(G.values)
        self.assertTrue(np.all(diag > 0))

    def test_gen_similarity_graph_default(self):
        G = gen_similarity_graph(self.X)

        self._basic_checks(G)
        self.assertEqual(set(G.index), set(self.X.index))
        self.assertEqual(set(G.columns), set(self.X.index))

    def test_gen_similarity_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_similarity_graph([1, 2, 3])

    def test_gen_correlation_graph_default(self):
        G = gen_correlation_graph(self.X, k=2, method="pearson")
        self._basic_checks(G)
        self.assertTrue((G.values >= 0).all())

    def test_gen_correlation_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_correlation_graph("not a df")

    def test_gen_threshold_graph_default(self):
        G = gen_threshold_graph(self.X, b=2.0, k=2)
        self._basic_checks(G)
        self.assertTrue((G.values >= 0).all())

    def test_gen_threshold_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_threshold_graph(None)

    def test_gen_gaussian_knn_graph_default(self):
        G = gen_gaussian_knn_graph(self.X, k=2)
        self._basic_checks(G)
        self.assertTrue((G.values >= 0).all())

    def test_gen_gaussian_knn_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_gaussian_knn_graph(123)

    def test_gen_lasso_graph_default(self):
        G = gen_lasso_graph(self.X, alpha=0.01)
        self._basic_checks(G)
        self.assertTrue((G.values >= 0).all())

    def test_gen_lasso_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_lasso_graph([1, 2, 3])

    def test_gen_mst_graph_default(self):
        G = gen_mst_graph(self.X)
        self._basic_checks(G)
        self.assertTrue((G.values >= 0).all())

    def test_gen_mst_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_mst_graph(5.0)

    def test_gen_snn_graph_default(self):
        G = gen_snn_graph(self.X, k=2)
        self._basic_checks(G)
        self.assertTrue((G.values >= 0).all())

    def test_gen_snn_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_snn_graph("oops, not a df")


if __name__ == "__main__":
    unittest.main()
