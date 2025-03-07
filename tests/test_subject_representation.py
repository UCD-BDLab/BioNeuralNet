import unittest
import pandas as pd
from bioneuralnet.subject_representation import GraphEmbedding

class TestGraphEmbedding(unittest.TestCase):

    def setUp(self):
        self.omics_data = pd.DataFrame(
            {"gene1": [1, 2, 3], "gene2": [4, 5, 6], "gene3": [7, 8, 9]},
            index=["sample1", "sample2", "sample3"],
        )
        self.phenotype_data = pd.DataFrame(
            {"phenotype": [0, 1, 2]}, index=["sample1", "sample2", "sample3"]
        )
        self.precomputed_embeddings = pd.DataFrame(
            {"dim1": [0.1, 0.2, 0.3], "dim2": [0.4, 0.5, 0.6], "dim3": [0.7, 0.8, 0.9]},
            index=["gene1", "gene2", "gene3"],
        )

    def test_run_with_precomputed_embeddings(self):
        graph_embed = GraphEmbedding(
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            embeddings=self.precomputed_embeddings,
            reduce_method="PCA",
            tune=False,
        )
        enhanced_omics_data = graph_embed.run()
        self.assertIsInstance(enhanced_omics_data, pd.DataFrame)
        self.assertEqual(enhanced_omics_data.shape[0], 3)

    def test_integrate_embeddings(self):
        node_embedding_values = pd.Series(
            {"gene1": 0.5, "gene2": 0.6, "gene3": 0.7},
            index=["gene1", "gene2", "gene3"],
        )
        with self.assertRaises(ValueError) as context:
            graph_embed = GraphEmbedding(
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                embeddings=None,
                reduce_method="PCA",
                tune=False,
            )
            graph_embed.integrate_embeddings(node_embedding_values)
        self.assertEqual(str(context.exception), "Embeddings must be provided as a pandas DataFrame.")

if __name__ == "__main__":
    unittest.main()
