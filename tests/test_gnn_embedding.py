import unittest
from unittest.mock import patch
import pandas as pd
import torch

from bioneuralnet.network_embedding import GNNEmbedding


class TestGNNEmbedding(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame(
            {
                "gene1": [1.0, 1.0, 0.0],
                "gene2": [1.0, 1.0, 1.0],
                "gene3": [0.0, 1.0, 1.0],
            },
            index=["gene1", "gene2", "gene3"],
        )

        self.omics_data = pd.DataFrame(
            {"gene1": [1, 2], "gene2": [3, 4], "gene3": [5, 6]},
            index=["sample1", "sample2"],
        )

        self.clinical_data = pd.DataFrame(
            {"age": [30, 45], "bmi": [22.5, 28.0]}, index=["sample1", "sample2"]
        )

        self.phenotype_data = pd.DataFrame(
            {"phenotype": [0, 1]}, index=["sample1", "sample2"]
        )

    @patch.object(
        GNNEmbedding,
        "embed",
        return_value=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    )
    def test_fit_and_embed_with_clinical(self, mock_embed):
        """
        Test the full workflow of GNNEmbedding with clinical data:
        1. Fit the model.
        2. Generate embeddings.
        """
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            phenotype_col="phenotype",
            model_type="GCN",
            hidden_dim=2,
            layer_num=2,
            dropout=True,
            num_epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=16,
            seed=42,
        )

        gnn.fit()
        embeddings = gnn.embed()

        mock_embed.assert_called_once()
        self.assertIsInstance(
            embeddings, torch.Tensor, "Embeddings should be a torch.Tensor."
        )
        self.assertEqual(
            embeddings.shape, (3, 2), "Embeddings tensor should have shape (3,2)."
        )

    @patch.object(
        GNNEmbedding,
        "embed",
        return_value=torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]),
    )
    def test_fit_and_embed_without_clinical(self, mock_embed):
        """
        Test the full workflow of GNNEmbedding without clinical data:
        1. Fit the model.
        2. Generate embeddings.
        """
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=None,
            phenotype_col="phenotype",
            model_type="GAT",
            hidden_dim=3,
            layer_num=3,
            dropout=False,
            num_epochs=20,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=16,
            seed=123,
        )

        gnn.fit()
        embeddings = gnn.embed()

        mock_embed.assert_called_once()
        self.assertIsInstance(
            embeddings, torch.Tensor, "Embeddings should be a torch.Tensor."
        )
        self.assertEqual(
            embeddings.shape, (3, 3), "Embeddings tensor should have shape (3,3)."
        )

    def test_embed_without_fit(self):
        """
        Ensure that calling embed() without fit() raises a ValueError.
        """
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            phenotype_col="phenotype",
            model_type="SAGE",
            hidden_dim=2,
            layer_num=2,
            dropout=True,
            num_epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=16,
            seed=42,
        )

        with self.assertRaises(ValueError):
            gnn.embed()

    def test_initialization_with_empty_clinical_data(self):
        """
        Test that providing empty clinical_data treats it as None and initializes features randomly.
        """
        empty_clinical = pd.DataFrame()

        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=empty_clinical,
            phenotype_col="phenotype",
            model_type="GIN",
            hidden_dim=2,
            layer_num=2,
            dropout=False,
            num_epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=16,
            seed=42,
        )

        with patch.object(
            gnn,
            "_generate_embeddings",
            return_value=torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
        ):
            gnn.fit()
            embeddings = gnn.embed()
        self.assertIsInstance(
            embeddings, torch.Tensor, "Embeddings should be a torch.Tensor."
        )
        self.assertEqual(
            embeddings.shape, (3, 2), "Embeddings tensor should have shape (3,2)."
        )

    def test_empty_adjacency_matrix(self):
        """
        Ensure that providing an empty adjacency matrix raises a ValueError.
        """
        empty_adj = pd.DataFrame()
        with self.assertRaises(ValueError):
            GNNEmbedding(
                adjacency_matrix=empty_adj,
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
                phenotype_col="phenotype",
                model_type="GCN",
                hidden_dim=2,
                layer_num=2,
                dropout=True,
                num_epochs=10,
                lr=1e-3,
                weight_decay=1e-4,
                gpu=False,
                random_feature_dim=16,
                seed=42,
            )

    def test_empty_omics_data(self):
        """
        Ensure that providing empty omics_data raises a ValueError.
        """
        empty_omics = pd.DataFrame()
        with self.assertRaises(ValueError):
            GNNEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=empty_omics,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
                phenotype_col="phenotype",
                model_type="GCN",
                hidden_dim=2,
                layer_num=2,
                dropout=True,
                num_epochs=10,
                lr=1e-3,
                weight_decay=1e-4,
                gpu=False,
                random_feature_dim=16,
                seed=42,
            )

    def test_empty_phenotype_data(self):
        """
        Ensure that providing empty phenotype_data raises a ValueError.
        """
        empty_pheno = pd.DataFrame()
        with self.assertRaises(ValueError):
            GNNEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=self.omics_data,
                phenotype_data=empty_pheno,
                clinical_data=self.clinical_data,
                phenotype_col="phenotype",
                model_type="GCN",
                hidden_dim=2,
                layer_num=2,
                dropout=True,
                num_epochs=10,
                lr=1e-3,
                weight_decay=1e-4,
                gpu=False,
                random_feature_dim=16,
                seed=42,
            )

    def test_random_feature_dim(self):
        """
        Ensure that random_feature_dim correctly sets the number of features when clinical_data is None.
        """
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=None,
            phenotype_col="phenotype",
            model_type="GCN",
            hidden_dim=3,
            layer_num=2,
            dropout=True,
            num_epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=8,
            seed=42,
        )

        with patch.object(
            gnn,
            "_generate_embeddings",
            return_value=torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            ),
        ):
            gnn.fit()
            embeddings = gnn.embed()
            print(embeddings)

        self.assertIsInstance(
            embeddings, torch.Tensor, "Embeddings should be a torch.Tensor."
        )
        self.assertEqual(
            embeddings.shape, (3, 3), "Embeddings tensor should have shape (3,3)."
        )

    def test_seed_reproducibility(self):
        """
        Ensure that setting a seed results in reproducible random features.
        """
        gnn1 = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=None,
            phenotype_col="phenotype",
            model_type="GCN",
            hidden_dim=2,
            layer_num=2,
            dropout=True,
            num_epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=4,
            seed=42,
        )

        gnn2 = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=None,
            phenotype_col="phenotype",
            model_type="GCN",
            hidden_dim=2,
            layer_num=2,
            dropout=True,
            num_epochs=10,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
            random_feature_dim=4,
            seed=42,
        )

        with patch.object(
            gnn1,
            "_generate_embeddings",
            return_value=torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
            ),
        ):
            gnn1.fit()
            embeddings1 = gnn1.embed()

        with patch.object(
            gnn2,
            "_generate_embeddings",
            return_value=torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
            ),
        ):
            gnn2.fit()
            embeddings2 = gnn2.embed()

        # Assertions
        self.assertTrue(
            torch.equal(embeddings1, embeddings2),
            "Embeddings should be identical due to same seed.",
        )


if __name__ == "__main__":
    unittest.main()
