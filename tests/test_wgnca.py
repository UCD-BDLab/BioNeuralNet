import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import subprocess
from bioneuralnet.graph_generation.wgcna import WGCNA

class TestWGCNA(unittest.TestCase):

    @patch('bioneuralnet.graph_generation.wgcna.subprocess.run')
    @patch('bioneuralnet.graph_generation.wgcna.WGCNA._create_output_dir')
    @patch('bioneuralnet.graph_generation.wgcna.WGCNA.preprocess_data')
    @patch('bioneuralnet.graph_generation.wgcna.WGCNA.load_global_network')
    def test_run_wgcna_success(self, mock_load_global, mock_preprocess, mock_create_output_dir, mock_subprocess):
        # Mock the output directory creation
        mock_create_output_dir.return_value = 'wgcna_output_1'

        # Mock the subprocess.run to simulate successful R script execution
        mock_subprocess.return_value = MagicMock(stdout='Success', stderr='')

        # Mock the load_global_network to return a dummy DataFrame
        dummy_df = pd.DataFrame({'Gene1': [1, 0], 'Gene2': [0, 1]}, index=['Gene1', 'Gene2'])
        mock_load_global.return_value = dummy_df

        wgcna = WGCNA(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/genes.csv', 'input/miRNA.csv'],
            data_types=['gene', 'miRNA'],
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
            output_dir='wgcna_output_1'
        )

        result = wgcna.run()

        # Assertions
        mock_create_output_dir.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_subprocess.assert_called_once()
        mock_load_global.assert_called_once_with()
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 2))
        self.assertListEqual(list(result.columns), ['Gene1', 'Gene2'])

    @patch('bioneuralnet.graph_generation.wgcna.subprocess.run')
    @patch('bioneuralnet.graph_generation.wgcna.WGCNA._create_output_dir')
    @patch('bioneuralnet.graph_generation.wgcna.WGCNA.preprocess_data')
    def test_run_wgcna_r_script_failure(self, mock_preprocess, mock_create_output_dir, mock_subprocess):
        # Mock the output directory creation
        mock_create_output_dir.return_value = 'wgcna_output_2'

        # Mock the subprocess.run to simulate R script execution failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd='Rscript WGCNA.R',
            stderr='Error executing R script'
        )

        wgcna = WGCNA(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/genes.csv', 'input/miRNA.csv'],
            data_types=['gene', 'miRNA'],
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
            output_dir='wgcna_output_2'
        )

        with self.assertRaises(subprocess.CalledProcessError):
            wgcna.run()

        # Assertions
        mock_create_output_dir.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_subprocess.assert_called_once()

    @patch('bioneuralnet.graph_generation.wgcna.pd.read_csv')
    @patch('bioneuralnet.graph_generation.wgcna.os.path.isfile')
    def test_load_global_network_success(self, mock_isfile, mock_read_csv):
        # Setup the mocks
        mock_isfile.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'Gene1': [1, 0], 'Gene2': [0, 1]}, index=['Gene1', 'Gene2'])

        wgcna = WGCNA(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/genes.csv', 'input/miRNA.csv'],
            data_types=['gene', 'miRNA'],
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
            output_dir='wgcna_output_1'
        )

        result = wgcna.load_global_network()

        # Assertions
        mock_isfile.assert_called_once_with('wgcna_output_1/global_network.csv')
        mock_read_csv.assert_called_once_with('wgcna_output_1/global_network.csv', index_col=0)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 2))

    @patch('bioneuralnet.graph_generation.wgcna.pd.read_csv')
    @patch('bioneuralnet.graph_generation.wgcna.os.path.isfile')
    def test_load_global_network_file_not_found(self, mock_isfile, mock_read_csv):
        # Setup the mocks
        mock_isfile.return_value = False

        wgcna = WGCNA(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/genes.csv', 'input/miRNA.csv'],
            data_types=['gene', 'miRNA'],
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
            output_dir='wgcna_output_1'
        )

        with self.assertRaises(FileNotFoundError):
            wgcna.load_global_network()

        # Assertions
        mock_isfile.assert_called_once_with('wgcna_output_1/global_network.csv')
        mock_read_csv.assert_not_called()

    @patch('bioneuralnet.graph_generation.wgcna.pd.read_csv')
    @patch('bioneuralnet.graph_generation.wgcna.os.path.isfile')
    def test_load_global_network_empty_csv(self, mock_isfile, mock_read_csv):
        # Setup the mocks
        mock_isfile.return_value = True
        mock_read_csv.side_effect = pd.errors.EmptyDataError

        wgcna = WGCNA(
            phenotype_file='input/phenotype_data.csv',
            omics_list=['input/genes.csv', 'input/miRNA.csv'],
            data_types=['gene', 'miRNA'],
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
            output_dir='wgcna_output_1'
        )

        with self.assertRaises(pd.errors.EmptyDataError):
            wgcna.load_global_network()

        # Assertions
        mock_isfile.assert_called_once_with('wgcna_output_1/global_network.csv')
        mock_read_csv.assert_called_once_with('wgcna_output_1/global_network.csv', index_col=0)

if __name__ == '__main__':
    unittest.main()
