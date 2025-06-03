import unittest
from unittest.mock import patch
import pandas as pd
import torch
from bioneuralnet.network_embedding import GNNEmbedding

class TestGNNEmbedding(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame([[1.0, 0.3, 0.1], [0.3, 1.0, 0.05], [0.1, 0.05, 1.0]],index=["gene1", "gene2", "gene3"],columns=["gene1", "gene2", "gene3"],)
        self.omics_data = pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4], "gene3": [5, 6]},index=["sample1", "sample2"])
        self.clinical_data = pd.DataFrame({"age": [30, 45], "bmi": [22.5, 28.0]}, index=["sample1", "sample2"])
        self.phenotype_data = pd.DataFrame({"phenotype": [0, 1]}, index=["sample1", "sample2"])

    @patch.object(GNNEmbedding,"embed",return_value=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    def test_fit_and_embed_with_clinical(self, mock_embed):
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            phenotype_col="phenotype",
            gpu=False,
            tune=False,
        )
        gnn.fit()
        embeddings = gnn.embed()
        mock_embed.assert_called_once()
        self.assertIsInstance(embeddings, torch.Tensor, "Embeddings should be a torch.Tensor.")
        self.assertEqual(embeddings.shape, (3, 2), "Embeddings tensor should have shape (3,2).")

    def test_embed_without_fit(self):
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            phenotype_col="phenotype",
        )
        with self.assertRaises(ValueError):
            gnn.embed()

    @patch.object(GNNEmbedding,"embed",return_value=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    def test_fit_and_embed_without_clinical(self, mock_embed):
        empty_clinical = pd.DataFrame()

        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=empty_clinical,
            phenotype_col="phenotype",
        )
        gnn.fit()
        embeddings = gnn.embed()

        mock_embed.assert_called_once()
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape, (3, 2))

    def test_empty_adjacency_matrix(self):
        empty_adj = pd.DataFrame()
        with self.assertRaises(ValueError):
            GNNEmbedding(
                adjacency_matrix=empty_adj,
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
                phenotype_col="phenotype",
            )

    def test_empty_omics_data(self):
        empty_omics = pd.DataFrame()
        with self.assertRaises(ValueError):
            GNNEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=empty_omics,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
                phenotype_col="phenotype",
            )

    def test_empty_phenotype_data(self):
        empty_pheno = pd.DataFrame()
        with self.assertRaises(ValueError):
            GNNEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=self.omics_data,
                phenotype_data=empty_pheno,
                clinical_data=self.clinical_data,
                phenotype_col="phenotype",
            )

if __name__ == "__main__":
    unittest.main()
