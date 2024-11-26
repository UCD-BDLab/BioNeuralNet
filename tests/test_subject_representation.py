import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from bioneuralnet.subject_representation.subject_representation import SubjectRepresentationEmbedding

class TestSubjectRepresentationEmbedding(unittest.TestCase):

    def setUp(self):
        # Create sample data
        self.adjacency_matrix = pd.DataFrame([[0,1,0],[1,0,1],[0,1,0]], index=['gene1','gene2','gene3'], columns=['gene1','gene2','gene3'])
        self.omics_data = pd.DataFrame({
            'gene1': [1, 2, 3],
            'gene2': [4, 5, 6],
            'gene3': [7, 8, 9]
        }, index=['sample1', 'sample2', 'sample3'])

        self.phenotype_data = pd.Series([0.1, 0.2, 0.3], index=['sample1', 'sample2', 'sample3'])

        self.clinical_data = pd.DataFrame({
            'age': [30, 40, 50],
            'bmi': [22.5, 27.8, 31.4]
        }, index=['sample1', 'sample2', 'sample3'])

    @patch('bioneuralnet.subject_representation.subject_representation.GNNEmbedding')
    @patch('bioneuralnet.subject_representation.subject_representation.Node2VecEmbedding')
    def test_generate_embeddings_gnns(self, mock_node2vec, mock_gnnembedding):
        # Mock GNNEmbedding
        mock_gnn = mock_gnnembedding.return_value
        mock_gnn.run.return_value = {'graph': pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], index=['gene1', 'gene2', 'gene3'])}

        subject_rep = SubjectRepresentationEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            embedding_method='GNNs'
        )
        node_features = pd.DataFrame({'feat1': [0.1, 0.2, 0.3]}, index=['gene1', 'gene2', 'gene3'])
        embeddings = subject_rep.generate_embeddings(node_features)
        mock_gnn.run.assert_called_once()
        self.assertEqual(embeddings.shape, (3, 2))

    @patch('bioneuralnet.subject_representation.subject_representation.GNNEmbedding')
    @patch('bioneuralnet.subject_representation.subject_representation.Node2VecEmbedding')
    def test_generate_embeddings_node2vec(self, mock_node2vec, mock_gnnembedding):
        # Mock Node2VecEmbedding
        mock_node2vec_instance = mock_node2vec.return_value
        mock_node2vec_instance.run.return_value = {'graph': pd.DataFrame({'node': ['gene1', 'gene2', 'gene3'], 'dim1': [0.1, 0.2, 0.3], 'dim2': [0.4, 0.5, 0.6]})}

        subject_rep = SubjectRepresentationEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            embedding_method='Node2Vec'
        )
        node_features = pd.DataFrame({'feat1': [0.1, 0.2, 0.3]}, index=['gene1', 'gene2', 'gene3'])
        embeddings = subject_rep.generate_embeddings(node_features)
        mock_node2vec_instance.run.assert_called_once()
        self.assertEqual(embeddings.shape, (3, 2))

    def test_run(self):
        # Mock methods to focus on the flow
        with patch.object(SubjectRepresentationEmbedding, 'compute_node_phenotype_correlation', return_value=pd.Series([0.1, 0.2, 0.3], index=['gene1', 'gene2', 'gene3'])) as mock_pheno_corr, \
             patch.object(SubjectRepresentationEmbedding, 'compute_node_clinical_correlation', return_value=pd.DataFrame({'feat1': [0.1, 0.2, 0.3]}, index=['gene1', 'gene2', 'gene3'])) as mock_clinical_corr, \
             patch.object(SubjectRepresentationEmbedding, 'generate_embeddings', return_value=pd.DataFrame({'dim1': [0.1, 0.2, 0.3]}, index=['gene1', 'gene2', 'gene3'])) as mock_generate_embeddings, \
             patch.object(SubjectRepresentationEmbedding, 'reduce_embeddings', return_value=pd.Series([0.1, 0.2, 0.3], index=['gene1', 'gene2', 'gene3'])) as mock_reduce_embeddings, \
             patch.object(SubjectRepresentationEmbedding, 'integrate_embeddings', return_value=self.omics_data) as mock_integrate_embeddings:

            subject_rep = SubjectRepresentationEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
                embedding_method='GNNs'
            )
            enhanced_omics_data = subject_rep.run()

            mock_pheno_corr.assert_called_once()
            mock_clinical_corr.assert_called_once()
            mock_generate_embeddings.assert_called_once()
            mock_reduce_embeddings.assert_called_once()
            mock_integrate_embeddings.assert_called_once()
            self.assertTrue(enhanced_omics_data.equals(self.omics_data))

if __name__ == '__main__':
    unittest.main()
