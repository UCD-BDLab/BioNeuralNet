import unittest
from unittest.mock import patch
import pandas as pd
import tempfile
import shutil
from bioneuralnet.downstream_task import DPMON
import warnings


class TestDPMON(unittest.TestCase):
    def setUp(self):
        self.adjacency_matrix = pd.DataFrame([[1.0, 0.3, 0.1], [0.3, 1.0, 0.05], [0.1, 0.05, 1.0]],index=["gene1", "gene2", "gene3"],columns=["gene1", "gene2", "gene3"],)
        self.tempfolder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tempfolder)

        self.omics_data1 = pd.DataFrame( {"gene1": [1, 2], "gene2": [3, 4]}, index=["sample1", "sample2"])
        self.omics_data2 = pd.DataFrame({"gene3": [5, 6]}, index=["sample1", "sample2"])
        self.phenotype_data = pd.DataFrame({"phenotype": [0, 1]}, index=["sample1", "sample2"])
        self.clinical_data = pd.DataFrame( {"age": [30, 45], "bmi": [22.5, 28.0]}, index=["sample1", "sample2"])

    @patch("bioneuralnet.downstream_task.dpmon.run_standard_training")
    def test_run_without_tune(self, mock_standard):
        warnings.filterwarnings("ignore")
        
        mock_standard.return_value = (pd.DataFrame({"Actual": [2, 3], "Predicted": [2, 2]}, index=["sample1", "sample2"]), 0.89)

        dpmon = DPMON(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=[self.omics_data1, self.omics_data2],
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            tune=False,
            gpu=False,
            output_dir=self.tempfolder
        )

        predictions = dpmon.run()
        mock_standard.assert_called_once()
        self.assertIsInstance(predictions, tuple)
        self.assertIsInstance(predictions[0], pd.DataFrame)
        self.assertIsInstance(predictions[1], float)

    @patch("bioneuralnet.downstream_task.dpmon.run_standard_training")
    @patch("bioneuralnet.downstream_task.dpmon.run_hyperparameter_tuning")
    def test_run_with_tune(self, mock_tune, mock_standard):
        warnings.filterwarnings("ignore")
        best_config_df = pd.DataFrame([{
            "gnn_hidden_dim": 64,
            "gnn_layer_num": 2,
            "nn_hidden_dim1": 128,
            "nn_hidden_dim2": 64,
            "num_epochs": 10,
        }])
        mock_tune.return_value = best_config_df
        mock_standard.return_value = (pd.DataFrame({"Actual": [0, 1], "Predicted": [0, 1]}), 0.95)

        dpmon = DPMON(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=[self.omics_data1, self.omics_data2],
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            tune=True,
            gpu=False,
            output_dir=self.tempfolder,
        )

        df, score = dpmon.run()
        mock_tune.assert_called_once()
        mock_standard.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(score, float)

    @patch("bioneuralnet.downstream_task.dpmon.run_standard_training")
    def test_empty_clinical_data(self, mock_run_standard_training):
        warnings.filterwarnings("ignore")
        mock_run_standard_training.return_value = None
        empty_clinical = pd.DataFrame()
        dpmon = DPMON(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=[self.omics_data1, self.omics_data2],
            phenotype_data=self.phenotype_data,
            clinical_data=empty_clinical,
            tune=False,
            gpu=False,
        )
        predictions = dpmon.run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    unittest.main()
