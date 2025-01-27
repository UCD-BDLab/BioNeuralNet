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

        self.clinical_data_df = pd.DataFrame(
            {"age": [30, 40, 50], "bmi": [22.5, 25.0, 28.0]},
            index=["sample1", "sample2", "sample3"],
        )

        self.phenotype_data = pd.Series(
            [0, 1, 2], index=["sample1", "sample2", "sample3"]
        )

        self.precomputed_embeddings = pd.DataFrame(
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
            {"dim1": [0.1, 0.2, 0.3], "dim2": [0.4, 0.5, 0.6], "dim3": [0.7, 0.8, 0.9]},
            index=["gene1", "gene2", "gene3"],
        ),
    )
    @patch.object(
        GraphEmbedding,
        "reduce_embeddings",
        return_value=pd.Series({"gene1": 0.5, "gene2": 0.6, "gene3": 0.7}),
    )
    @patch.object(
        GraphEmbedding,
        "integrate_embeddings",
        return_value=pd.DataFrame(
            {
                "gene1": [1.0 * 0.5, 2.0 * 0.5, 3.0 * 0.5],
                "gene2": [4.0 * 0.6, 5.0 * 0.6, 6.0 * 0.6],
                "gene3": [7.0 * 0.7, 8.0 * 0.7, 9.0 * 0.7],
            },
            index=["sample1", "sample2", "sample3"],
        ),
    )
    def test_run_with_clinical_and_no_precomputed_embeddings(
        self, mock_integrate, mock_reduce, mock_generate
    ):
        """
        Test that GraphEmbedding.run() returns an expected DataFrame
        and calls underlying steps when clinical data is provided and no precomputed embeddings.
        """
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
            reduce_method="PCA",
        )

        enhanced_omics_data = graph_embed.run()

        self.assertIsInstance(
            enhanced_omics_data, pd.DataFrame, "Output should be a pandas DataFrame."
        )
        self.assertEqual(
            enhanced_omics_data.shape,
            (3, 3),
            "Output shape should match expected shape (3,3).",
        )
        self.assertListEqual(
            list(enhanced_omics_data.columns),
            ["gene1", "gene2", "gene3"],
            "Columns should match the integrated omics features.",
        )

        # Ensure that the methods were called once
        mock_generate.assert_called_once()
        mock_reduce.assert_called_once_with(mock_generate.return_value)
        mock_integrate.assert_called_once_with(mock_reduce.return_value)

    @patch.object(
        GraphEmbedding,
        "generate_embeddings",
        return_value=pd.DataFrame(
            {"dim1": [0.1, 0.2, 0.3], "dim2": [0.4, 0.5, 0.6], "dim3": [0.7, 0.8, 0.9]},
            index=["gene1", "gene2", "gene3"],
        ),
    )
    @patch.object(
        GraphEmbedding,
        "reduce_embeddings",
        return_value=pd.Series({"gene1": 0.5, "gene2": 0.6, "gene3": 0.7}),
    )
    @patch.object(
        GraphEmbedding,
        "integrate_embeddings",
        return_value=pd.DataFrame(
            {
                "gene1": [1.0 * 0.5, 2.0 * 0.5, 3.0 * 0.5],
                "gene2": [4.0 * 0.6, 5.0 * 0.6, 6.0 * 0.6],
                "gene3": [7.0 * 0.7, 8.0 * 0.7, 9.0 * 0.7],
                "finalgold_visit": [0, 1, 2],
            },
            index=["sample1", "sample2", "sample3"],
        ),
    )
    def test_run_with_precomputed_embeddings(
        self, mock_integrate, mock_reduce, mock_generate
    ):
        """
        Test that GraphEmbedding.run() returns an expected DataFrame
        and calls underlying steps when precomputed embeddings are provided.
        """
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=self.precomputed_embeddings,
            reduce_method="AVG",
        )

        enhanced_omics_data = graph_embed.run()

        self.assertIsInstance(
            enhanced_omics_data, pd.DataFrame, "Output should be a pandas DataFrame."
        )
        self.assertEqual(
            enhanced_omics_data.shape,
            (3, 4),
            "Output shape should match expected shape (3,4).",
        )
        self.assertListEqual(
            list(enhanced_omics_data.columns),
            ["gene1", "gene2", "gene3", "finalgold_visit"],
            "Columns should match the integrated omics features.",
        )

        mock_generate.assert_called_once()
        mock_reduce.assert_called_once_with(mock_generate.return_value)
        mock_integrate.assert_called_once_with(mock_reduce.return_value)

    @patch.object(GraphEmbedding, "reduce_embeddings")
    def test_reduce_embeddings_avg(self, mock_reduce):
        """
        Test that reduce_embeddings works with the 'AVG' method.
        """
        mock_reduce.return_value = pd.Series({"gene1": 0.4, "gene2": 0.5, "gene3": 0.6})
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
            reduce_method="AVG",
        )
        embeddings_df = pd.DataFrame(
            {"dim1": [0.1, 0.2, 0.3], "dim2": [0.4, 0.5, 0.6], "dim3": [0.7, 0.8, 0.9]},
            index=["gene1", "gene2", "gene3"],
        )
        result = graph_embed.reduce_embeddings(embeddings_df)

        self.assertIsInstance(result, pd.Series)
        self.assertListEqual(
            list(result), [0.4, 0.5, 0.6], "Result should match mocked values."
        )
        mock_reduce.assert_called_once_with(embeddings_df)

    @patch.object(GraphEmbedding, "reduce_embeddings")
    def test_reduce_embeddings_max(self, mock_reduce):
        """
        Test that reduce_embeddings works with the 'MAX' method.
        """
        mock_reduce.return_value = pd.Series({"gene1": 0.7, "gene2": 0.8, "gene3": 0.9})
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
            reduce_method="MAX",
        )
        embeddings_df = pd.DataFrame(
            {"dim1": [0.1, 0.2, 0.3], "dim2": [0.4, 0.5, 0.6], "dim3": [0.7, 0.8, 0.9]},
            index=["gene1", "gene2", "gene3"],
        )
        result = graph_embed.reduce_embeddings(embeddings_df)

        self.assertIsInstance(result, pd.Series)
        self.assertListEqual(
            list(result), [0.7, 0.8, 0.9], "Result should match mocked values."
        )
        mock_reduce.assert_called_once_with(embeddings_df)

    @patch.object(GraphEmbedding, "integrate_embeddings")
    def test_integrate_embeddings(self, mock_integrate):
        """
        Test that integrate_embeddings correctly integrates the reduced embeddings.
        """
        node_embedding_values = pd.Series({"gene1": 0.5, "gene2": 0.6, "gene3": 0.7})
        expected_output = pd.DataFrame(
            {
                "gene1": [1.0 * 0.5, 2.0 * 0.5, 3.0 * 0.5],
                "gene2": [4.0 * 0.6, 5.0 * 0.6, 6.0 * 0.6],
                "gene3": [7.0 * 0.7, 8.0 * 0.7, 9.0 * 0.7],
            },
            index=["sample1", "sample2", "sample3"],
        )
        mock_integrate.return_value = expected_output

        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data_df,
            embeddings=None,
            reduce_method="PCA",
        )

        result = graph_embed.integrate_embeddings(node_embedding_values)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 3))
        self.assertListEqual(
            list(result.columns),
            ["gene1", "gene2", "gene3"],
            "Columns should match the integrated omics features.",
        )
        mock_integrate.assert_called_once_with(node_embedding_values)


if __name__ == "__main__":
    unittest.main()
