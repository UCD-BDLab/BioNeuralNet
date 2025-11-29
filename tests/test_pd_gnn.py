import unittest
import pandas as pd
import numpy as np
import torch
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

from models.gnn.gnn_trainer import GNNTrainer, GNNResults, train_gnn_pd


@unittest.skipIf(not PYG_AVAILABLE, "PyTorch Geometric not available")
class TestPDGNN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        if not PYG_AVAILABLE:
            self.skipTest("PyTorch Geometric not available")

        # Create small synthetic graph
        n_nodes = 20
        n_features = 4

        # Node features
        x = torch.randn(n_nodes, n_features)

        # Create edge index (ring graph)
        edge_list = []
        for i in range(n_nodes):
            edge_list.append([i, (i + 1) % n_nodes])
            edge_list.append([(i + 1) % n_nodes, i])  # undirected

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        self.graph_data = Data(x=x, edge_index=edge_index)

    def test_gnn_trainer_initialization(self):
        """Test GNN trainer initialization."""
        trainer = GNNTrainer(
            model_type="GCN",
            hidden_dim=32,
            layer_num=2,
            device="cpu",
        )

        self.assertEqual(trainer.model_type, "GCN")
        self.assertEqual(trainer.hidden_dim, 32)
        self.assertIsNone(trainer.model)  # Not trained yet

    def test_gnn_trainer_train(self):
        """Test GNN training."""
        trainer = GNNTrainer(
            model_type="GCN",
            hidden_dim=16,
            layer_num=2,
            num_epochs=5,  # Short training for test
            device="cpu",
        )

        trainer.train(self.graph_data, supervision_targets=None)
        self.assertIsNotNone(trainer.model)
        self.assertGreater(len(trainer.training_losses), 0)

    def test_gnn_trainer_get_embeddings(self):
        """Test embedding generation."""
        trainer = GNNTrainer(
            model_type="GCN",
            hidden_dim=16,
            layer_num=2,
            num_epochs=5,
            device="cpu",
        )

        trainer.train(self.graph_data)
        embeddings = trainer.get_embeddings(self.graph_data)

        self.assertEqual(embeddings.shape[0], self.graph_data.num_nodes)
        self.assertEqual(embeddings.shape[1], 16)  # hidden_dim

    def test_train_gnn_pd_convenience(self):
        """Test train_gnn_pd convenience function."""
        trainer, results = train_gnn_pd(
            self.graph_data,
            model_type="GCN",
            hidden_dim=16,
            layer_num=2,
            num_epochs=5,
            device="cpu",
        )

        self.assertIsInstance(trainer, GNNTrainer)
        self.assertIsInstance(results, GNNResults)
        self.assertEqual(results.embeddings.shape[0], self.graph_data.num_nodes)

    def test_different_model_types(self):
        """Test different GNN model types."""
        for model_type in ["GCN", "SAGE"]:  # Test GCN and SAGE
            trainer = GNNTrainer(
                model_type=model_type,
                hidden_dim=16,
                layer_num=2,
                num_epochs=3,
                device="cpu",
            )
            trainer.train(self.graph_data)
            embeddings = trainer.get_embeddings(self.graph_data)

            self.assertEqual(embeddings.shape[0], self.graph_data.num_nodes)


if __name__ == "__main__":
    unittest.main()
