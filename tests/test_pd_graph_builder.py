import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
pd_notebooks = project_root / "PD-Notebooks"
if str(pd_notebooks) not in sys.path:
    sys.path.insert(0, str(pd_notebooks))

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from graph_builder import (
    build_correlation_graph,
    adjacency_to_pyg,
    build_pd_graph,
    GraphData,
)


@unittest.skipIf(not PYG_AVAILABLE, "PyTorch Geometric not available")
class TestPDGraphBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic expression data
        np.random.seed(42)
        n_genes = 50
        n_samples = 10

        # Create correlated expression data
        base_expr = np.random.randn(n_samples)
        self.expression_df = pd.DataFrame(
            np.array([base_expr + np.random.randn(n_samples) * 0.1 for _ in range(n_genes)]).T,
            index=[f"SAMPLE_{i}" for i in range(n_samples)],
            columns=[f"GENE_{i}" for i in range(n_genes)],
        ).T  # genes × samples

        # Create node features
        self.node_features = pd.DataFrame(
            np.random.randn(n_genes, 2),
            index=self.expression_df.index,
            columns=["mean", "variance"],
        )

    def test_build_correlation_graph(self):
        """Test correlation graph construction."""
        adjacency = build_correlation_graph(
            self.expression_df,
            method="pearson",
            threshold=0.5,
            use_abs=True,
        )

        self.assertEqual(adjacency.shape[0], self.expression_df.shape[0])
        self.assertEqual(adjacency.shape[1], self.expression_df.shape[0])
        self.assertTrue((adjacency >= 0).all().all())  # Non-negative
        self.assertTrue((adjacency <= 1).all().all())  # Correlations ≤ 1

    def test_adjacency_to_pyg(self):
        """Test conversion to PyTorch Geometric format."""
        # Create small adjacency matrix
        adj = pd.DataFrame(
            [[1.0, 0.8, 0.0], [0.8, 1.0, 0.7], [0.0, 0.7, 1.0]],
            index=["G1", "G2", "G3"],
            columns=["G1", "G2", "G3"],
        )

        node_feat = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            index=["G1", "G2", "G3"],
            columns=["f1", "f2"],
        )

        data, node_names = adjacency_to_pyg(adj, node_feat)

        self.assertIsInstance(data, Data)
        self.assertEqual(data.num_nodes, 3)
        self.assertEqual(data.num_node_features, 2)
        self.assertGreater(data.num_edges, 0)

    def test_build_pd_graph(self):
        """Test complete graph building pipeline."""
        graph_data = build_pd_graph(
            self.expression_df,
            self.node_features,
            threshold=0.5,
            use_bioneuralnet=False,
        )

        self.assertIsInstance(graph_data, GraphData)
        self.assertIsInstance(graph_data.data, Data)
        self.assertIsInstance(graph_data.adjacency_matrix, pd.DataFrame)
        self.assertEqual(graph_data.data.num_nodes, self.expression_df.shape[0])


if __name__ == "__main__":
    unittest.main()
