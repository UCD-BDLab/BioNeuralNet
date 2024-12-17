import unittest
from unittest.mock import patch
import pandas as pd
from bioneuralnet.integrated_tasks import DPMON

class TestDPMON(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame(
            [[1.0, 0.3, 0.1],
             [0.3, 1.0, 0.05],
             [0.1, 0.05, 1.0]],
            index=['gene1', 'gene2', 'gene3'],
            columns=['gene1', 'gene2', 'gene3']
        )

        self.omics_data1 = pd.DataFrame({
            'gene1': [1,2],
            'gene2': [3,4]
        }, index=['sample1', 'sample2'])

        self.omics_data2 = pd.DataFrame({
            'gene3': [5,6]
        }, index=['sample1', 'sample2'])

        self.phenotype_data = pd.DataFrame({
            'finalgold_visit': [2,3]
        }, index=['sample1', 'sample2'])

        self.features_data = pd.DataFrame({
            'age': [30,45],
            'bmi': [22.5,28.0]
        }, index=['sample1', 'sample2'])

    @patch('bioneuralnet.integrated_tasks.DPMON.run_standard_training')
    @patch('bioneuralnet.integrated_tasks.DPMON.GnnEmbedding')
    def test_run_without_tune(self, mock_gnn, mock_standard):
        # Mock GnnEmbedding.run to return embeddings
        mock_gnn_instance = mock_gnn.return_value
        mock_gnn_instance.run.return_value = {'graph': pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], 
                                                                    index=['gene1', 'gene2', 'gene3'])}

        # Mock run_standard_training to return predictions
        mock_standard.return_value = pd.DataFrame({
            'Actual': [2, 3],
            'Predicted': [2, 2]
        }, index=['sample1', 'sample2'])

        dpmon = DPMON(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=[self.omics_data1, self.omics_data2],
            phenotype_file=self.phenotype_data,
            features_file=self.features_data,
            model='GCN',
            tune=False,
            gpu=False,
            output_dir='test_output'
        )

        predictions = dpmon.run()
        mock_gnn_instance.run.assert_called_once()
        mock_standard.assert_called_once()
        self.assertIn('Actual', predictions.columns)
        self.assertIn('Predicted', predictions.columns)
        self.assertEqual(predictions.shape, (2,2))

    @patch('bioneuralnet.integrated_tasks.DPMON.run_hyperparameter_tuning')
    def test_run_with_tune(self, mock_tune):
        dpmon = DPMON(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=[self.omics_data1, self.omics_data2],
            phenotype_file=self.phenotype_data,
            features_file=self.features_data,
            model='GAT',
            tune=True,
            gpu=False
        )

        predictions = dpmon.run()
        mock_tune.assert_called_once()
        self.assertTrue(predictions.empty)

    def test_missing_clinical_data(self):
        with self.assertRaises(ValueError):
            DPMON(
                adjacency_matrix=self.adjacency_matrix,
                omics_list=[self.omics_data1, self.omics_data2],
                phenotype_file=self.phenotype_data,
                features_file=None,  
                model='SAGE',
                tune=False
            )

if __name__ == '__main__':
    unittest.main()
