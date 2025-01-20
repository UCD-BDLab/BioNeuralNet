import unittest
from unittest.mock import patch
import pandas as pd

from bioneuralnet.subject_representation import GraphEmbedding


class TestGraphEmbedding(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            index=["gene1", "gene2", "gene3"],
            columns=["gene1", "gene2", "gene3"],
        )

        self.omics_data = pd.DataFrame(
            {"gene1": [1, 2, 3], "gene2": [4, 5, 6], "gene3": [7, 8, 9]},
            index=["sample1", "sample2", "sample3"],
        )

        self.omics_data["finalgold_visit"] = [0, 1, 2]

        self.clinical_data_df = pd.DataFrame(
            {"age": [30, 40, 50]}, index=["sample1", "sample2", "sample3"]
        )

        self.phenotype_data = pd.Series(
            [0, 1, 2], index=["sample1", "sample2", "sample3"]
        )

        self.embeddings = pd.DataFrame(
            {
                "dim1": [0.1, 0.2, 0.3],
                "dim2": [0.4, 0.5, 0.6],
                "dim3": [0.7, 0.8, 0.9],
            },
            index=["gene1", "gene2", "gene3"],
        )

    @patch.object(
        GraphEmbedding,
        "generate_embeddings",
        return_value=pd.DataFrame(
            {"dim1": [0.1, 0.2, 0.3]}, index=["gene1", "gene2", "gene3"]
        ),
    )
    @patch.object(
        GraphEmbedding,
        "reduce_embeddings",
        return_value=pd.Series({"gene1": 0.1, "gene2": 0.2, "gene3": 0.3}),
    )
    @patch.object(
        GraphEmbedding,
        "integrate_embeddings",
        return_value=pd.DataFrame(
            {
                "gene1": [1.1, 2.2, 3.3],
                "gene2": [4.4, 5.5, 6.6],
                "gene3": [7.7, 8.8, 9.9],
                "finalgold_visit": [0, 1, 2],
            },
            index=["sample1", "sample2", "sample3"],
        ),
    )
    def test_run(self, mock_integrate, mock_reduce, mock_generate):
        """
        Test that GraphEmbedding.run() returns an expected DataFrame
        and calls underlying steps.
        """
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
        )

        enhanced_omics_data = graph_embed.run()

        self.assertIsInstance(
            enhanced_omics_data,
            pd.DataFrame,
        )
        self.assertEqual(
            enhanced_omics_data.shape,
            (3, 4),
            "Output shape should match expected shape (3,4).",
        )
        self.assertListEqual(
            list(enhanced_omics_data.columns),
            ["gene1", "gene2", "gene3", "finalgold_visit"],
            "Columns should match the integrated omics + finalgold_visit.",
        )

        mock_generate.assert_called_once()
        mock_reduce.assert_called_once()
        mock_integrate.assert_called_once()

    @patch.object(GraphEmbedding, "reduce_embeddings")
    def test_reduce_embeddings_average(self, mock_reduce):
        """
        Test that reduce_embeddings works with the 'average' method.
        """
        mock_reduce.return_value = pd.Series({"gene1": 0.4, "gene2": 0.5, "gene3": 0.6})
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
        )
        result = graph_embed.reduce_embeddings(self.embeddings, method="average")

        self.assertIsInstance(result, pd.Series)
        self.assertListEqual(
            result.tolist(), [0.4, 0.5, 0.6], "Result should match mocked values."
        )
        mock_reduce.assert_called_once_with(self.embeddings, method="average")

    @patch.object(GraphEmbedding, "reduce_embeddings")
    def test_reduce_embeddings_maximum(self, mock_reduce):
        """
        Test that reduce_embeddings works with the 'maximum' method.
        """
        mock_reduce.return_value = pd.Series({"gene1": 0.7, "gene2": 0.8, "gene3": 0.9})
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
        )
        result = graph_embed.reduce_embeddings(self.embeddings, method="maximum")

        self.assertIsInstance(result, pd.Series)
        self.assertListEqual(
            result.tolist(), [0.7, 0.8, 0.9], "Result should match mocked values."
        )
        mock_reduce.assert_called_once_with(self.embeddings, method="maximum")


if __name__ == "__main__":
    unittest.main()
