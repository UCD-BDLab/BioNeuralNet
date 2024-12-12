import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from bioneuralnet.graph_generation.smccnet import SmCCNet
import subprocess

class TestSmCCNet(unittest.TestCase):

    @patch('bioneuralnet.graph_generation.smccnet.subprocess.run')
    @patch('bioneuralnet.graph_generation.smccnet.SmCCNet._create_output_dir')
    @patch('bioneuralnet.graph_generation.smccnet.SmCCNet.preprocess_data')
    @patch('bioneuralnet.graph_generation.smccnet.SmCCNet.load_global_network')
    def test_run_smccnet_success(self, mock_load_global, mock_preprocess, mock_create_output_dir, mock_subprocess):
        # Mock the output directory creation
        mock_create_output_dir.return_value = 'smccnet_output_1'

        # Mock the subprocess.run to simulate successful R script execution
        mock_subprocess.return_value = MagicMock(stdout='Success', stderr='')

        # Mock the load_global_network to return a dummy DataFrame
        dummy_df = pd.DataFrame({'Gene1': [1, 0], 'Gene2': [0, 1]}, index=['Gene1', 'Gene2'])
        mock_load_global.return_value = dummy_df

        smccnet = SmCCNet(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/proteins.csv', 'input/metabolites.csv'],
            data_types=['protein', 'metabolite'],
            kfold=5,
            summarization='PCA',
            seed=732,
        )

        result = smccnet.run()

        # Assertions
        mock_create_output_dir.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_subprocess.assert_called_once()
        mock_load_global.assert_called_once_with('smccnet_output_1')
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 2))
        self.assertListEqual(list(result.columns), ['Gene1', 'Gene2'])

    @patch('bioneuralnet.graph_generation.smccnet.subprocess.run')
    @patch('bioneuralnet.graph_generation.smccnet.SmCCNet._create_output_dir')
    @patch('bioneuralnet.graph_generation.smccnet.SmCCNet.preprocess_data')
    def test_run_smccnet_r_script_failure(self, mock_preprocess, mock_create_output_dir, mock_subprocess):
        # Mock the output directory creation
        mock_create_output_dir.return_value = 'smccnet_output_2'

        # Mock the subprocess.run to simulate R script execution failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd='Rscript SmCCNet.R',
            stderr='Error executing R script'
        )

        smccnet = SmCCNet(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/proteins.csv', 'input/metabolites.csv'],
            data_types=['protein', 'metabolite']
        )

        with self.assertRaises(subprocess.CalledProcessError):
            smccnet.run()

        # Assertions
        mock_create_output_dir.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_subprocess.assert_called_once()

    @patch('bioneuralnet.graph_generation.smccnet.pd.read_csv')
    @patch('bioneuralnet.graph_generation.smccnet.os.path.isfile')
    def test_load_global_network_success(self, mock_isfile, mock_read_csv):
        """
        Test that SmCCNet.run() successfully returns a DataFrame if the R script executes without errors.
        """
        
        # Setup the mocks
        mock_isfile.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'Gene1': [1, 0], 'Gene2': [0, 1]}, index=['Gene1', 'Gene2'])

        smccnet = SmCCNet(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/proteins.csv', 'input/metabolites.csv'],
            data_types=['protein', 'metabolite']
        )

        result = smccnet.load_global_network('smccnet_output_1')

        # Assertions
        mock_isfile.assert_called_once_with('smccnet_output_1/global_network.csv')
        mock_read_csv.assert_called_once_with('smccnet_output_1/global_network.csv', index_col=0)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 2))

    @patch('bioneuralnet.graph_generation.smccnet.pd.read_csv')
    @patch('bioneuralnet.graph_generation.smccnet.os.path.isfile')
    def test_load_global_network_file_not_found(self, mock_isfile, mock_read_csv):
        # Setup the mocks
        mock_isfile.return_value = False

        smccnet = SmCCNet(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/proteins.csv', 'input/metabolites.csv'],
            data_types=['protein', 'metabolite']
        )

        with self.assertRaises(FileNotFoundError):
            smccnet.load_global_network('smccnet_output_1')

        # Assertions
        mock_isfile.assert_called_once_with('smccnet_output_1/global_network.csv')
        mock_read_csv.assert_not_called()

    @patch('bioneuralnet.graph_generation.smccnet.pd.read_csv')
    @patch('bioneuralnet.graph_generation.smccnet.os.path.isfile')
    def test_load_global_network_empty_csv(self, mock_isfile, mock_read_csv):
        # Setup the mocks
        mock_isfile.return_value = True
        mock_read_csv.side_effect = pd.errors.EmptyDataError

        smccnet = SmCCNet(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/proteins.csv', 'input/metabolites.csv'],
            data_types=['protein', 'metabolite']
        )

        with self.assertRaises(pd.errors.EmptyDataError):
            smccnet.load_global_network('smccnet_output_1')

        # Assertions
        mock_isfile.assert_called_once_with('smccnet_output_1/global_network.csv')
        mock_read_csv.assert_called_once_with('smccnet_output_1/global_network.csv', index_col=0)

if __name__ == '__main__':
    unittest.main()
