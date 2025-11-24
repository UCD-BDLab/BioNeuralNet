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
from bioneuralnet.datasets import DatasetLoader


class TestUtilsGraph(unittest.TestCase):
    def setUp(self):
        example = DatasetLoader("example")
        X1 = example["X1"]
        X2 = example["X2"]

        if not isinstance(X1, pd.DataFrame):
            raise TypeError("X1 from example dataset must be a DataFrame")
        if not isinstance(X2, pd.DataFrame):
            raise TypeError("X2 from example dataset must be a DataFrame")

        # A small subset of features to keep tests light
        X1_small = X1.iloc[:, 0:15]
        X2_small = X2.iloc[:, 0:15]

        X_concat = pd.concat([X1_small, X2_small], axis=1)
        self.X = X_concat
        self.N = self.X.shape[1]

    def _basic_checks(self, G: pd.DataFrame):
        self.assertIsInstance(G, pd.DataFrame)
        self.assertEqual(G.shape, (self.N, self.N))

        expected_cols = list(self.X.columns)
        self.assertEqual(list(G.index), expected_cols)
        self.assertEqual(list(G.columns), expected_cols)

        values = G.values
        self.assertTrue(np.all(values >= 0))

        row_sums = G.sum(axis=1).values
        close_to_one = np.isclose(row_sums, 1.0, atol=1e-5)
        close_to_zero = np.isclose(row_sums, 0.0, atol=1e-5)
        is_normalized = close_to_one | close_to_zero
        self.assertTrue(is_normalized.all())

    def test_gen_similarity_graph_default(self):
        G = gen_similarity_graph(self.X, k=3)
        self._basic_checks(G)

    def test_gen_similarity_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_similarity_graph(np.array([[1, 2], [3, 4]]))

    def test_gen_correlation_graph_default(self):
        G = gen_correlation_graph(
            self.X,
            k=3,
            method="pearson",
            signed=True,
            normalize=True,
            mutual=False,
            per_node=True,
            threshold=None,
            self_loops=False,
        )
        self._basic_checks(G)

    def test_gen_correlation_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_correlation_graph("not a df")

    def test_gen_threshold_graph_default(self):
        G = gen_threshold_graph(
            self.X,
            b=4.0,
            k=3,
            mutual=False,
            self_loops=False,
            normalize=True,
        )
        self._basic_checks(G)

    def test_gen_threshold_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_threshold_graph(None)

    def test_gen_gaussian_knn_graph_default(self):
        G = gen_gaussian_knn_graph(
            self.X,
            k=3,
            mutual=False,
            self_loops=True,
            normalize=True,
        )
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
        G = gen_snn_graph(self.X, k=3)
        self._basic_checks(G)

    def test_gen_snn_graph_type_error(self):
        with self.assertRaises(TypeError):
            gen_snn_graph("oops, not a df")


if __name__ == "__main__":
    unittest.main()
