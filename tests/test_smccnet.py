import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from bioneuralnet.external_tools import SmCCNet
import subprocess


class TestSmCCNet(unittest.TestCase):

    def setUp(self):
        self.phenotype_df = pd.DataFrame(
            {
                "ID": ["S1", "S2", "S3", "S4"],
                "Phenotype": [0, 1, 0, 1],
            }
        )
        self.omics_df1 = pd.DataFrame(
            {
                "ID": ["S1", "S2", "S3", "S4"],
                "GeneA": [1.2, 2.3, 3.1, 4.0],
                "GeneB": [2.1, 3.4, 1.2, 3.3],
                "GeneC": [3.3, 1.5, 2.2, 4.1],
            }
        )
        self.omics_df2 = pd.DataFrame(
            {
                "ID": ["S1", "S2", "S3", "S4"],
                "ProtD": [4.2, 5.3, 6.1, 7.0],
                "ProtE": [5.1, 6.4, 4.2, 6.3],
                "ProtF": [6.3, 4.5, 5.2, 7.1],
            }
        )

        self.omics_dfs = [self.omics_df1, self.omics_df2]
        self.data_types = ["Genomics", "Proteomics"]

    @patch("bioneuralnet.external_tools.smccnet.pd.read_csv")
    @patch("bioneuralnet.external_tools.smccnet.subprocess.run")
    def test_smccnet_successful_run(self, mock_run, mock_read_csv):
        mock_completed_process = MagicMock()
        mock_completed_process.returncode = 0
        mock_run.return_value = mock_completed_process

        mock_adjacency = pd.DataFrame(
            [
                [1.0, 0.8, 0.5, 0.2, 0.3, 0.4],
                [0.8, 1.0, 0.6, 0.3, 0.4, 0.5],
                [0.5, 0.6, 1.0, 0.4, 0.5, 0.6],
                [0.2, 0.3, 0.4, 1.0, 0.7, 0.4],
                [0.3, 0.4, 0.5, 0.7, 1.0, 0.5],
                [0.4, 0.5, 0.6, 0.4, 0.5, 1.0],
            ],
            index=["GeneA", "GeneB", "GeneC", "ProtD", "ProtE", "ProtF"],
            columns=["GeneA", "GeneB", "GeneC", "ProtD", "ProtE", "ProtF"],
        )
        mock_read_csv.return_value = mock_adjacency

        smccnet = SmCCNet(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            kfold=5,
            summarization="PCA",
            seed=732,
        )
        adjacency_matrix = smccnet.run()
        self.assertIsInstance(adjacency_matrix, pd.DataFrame)
        self.assertFalse(adjacency_matrix.isnull().values.any())
        self.assertEqual(adjacency_matrix.shape, (6, 6))
        self.assertListEqual(
            list(adjacency_matrix.columns),
            ["GeneA", "GeneB", "GeneC", "ProtD", "ProtE", "ProtF"],
        )
        self.assertListEqual(
            list(adjacency_matrix.index),
            ["GeneA", "GeneB", "GeneC", "ProtD", "ProtE", "ProtF"],
        )
        self.assertAlmostEqual(adjacency_matrix.loc["GeneA", "GeneB"], 0.8)
        self.assertAlmostEqual(adjacency_matrix.loc["ProtD", "ProtF"], 0.4)

    @patch("bioneuralnet.external_tools.smccnet.subprocess.run")
    def test_smccnet_run_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="Rscript SmCCNet.R",
            stderr='Some Error"\nExecution halted',
        )

        smccnet = SmCCNet(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            kfold=5,
            summarization="PCA",
            seed=732,
        )

        with self.assertRaises(subprocess.CalledProcessError):
            smccnet.run()

    def test_mismatched_omics_and_data_types(self):
        with self.assertRaises(ValueError):
            SmCCNet(
                phenotype_df=self.phenotype_df,
                omics_dfs=self.omics_dfs,
                data_types=["Transcriptomics"],
                kfold=5,
                summarization="PCA",
                seed=732,
            )

    @patch("bioneuralnet.external_tools.smccnet.subprocess.run")
    def test_no_valid_samples(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="Rscript SmCCNet.R",
            stderr="Error: No valid samples after preprocessing.\nExecution halted",
        )

        self.omics_dfs[0].iloc[0, 1] = pd.NA
        self.omics_dfs[1].iloc[1, 2] = pd.NA

        smccnet = SmCCNet(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            kfold=5,
            summarization="PCA",
            seed=732,
        )

        with self.assertRaises(subprocess.CalledProcessError):
            smccnet.run()

    # NOT sure how to effectively test single omics data yet

    # @patch("bioneuralnet.external_tools.smccnet.subprocess.run")
    # def test_smccnet_single_omics(self, mock_run):
    #     """
    #     Test SmCCNet with single-omics data. Simulates single-omics mode by
    #     verifying the adjacency matrix returned by the R script's JSON output.
    #     """
    #     mock_completed_process = MagicMock()
    #     mock_completed_process.returncode = 0
    #     mock_run.return_value = mock_completed_process

    #     single_omics_df = pd.DataFrame(
    #         {
    #             "ID": ["S1", "S2", "S3", "S4"],
    #             "GeneA": [1.2, 2.3, 3.1, 4.0],
    #             "GeneB": [2.1, 3.4, 1.2, 3.3],
    #             "GeneC": [3.3, 1.5, 2.2, 4.1],
    #         }
    #     )

    #     smccnet = SmCCNet(
    #         phenotype_df=self.phenotype_df,
    #         omics_dfs=[single_omics_df],
    #         data_types=["Transcriptomics"],
    #         kfold=5,
    #         summarization="PCA",
    #         seed=732,
    #     )

    #     adjacency_matrix = smccnet.run()

    #     self.assertIsInstance(adjacency_matrix, pd.DataFrame)

    #     self.assertEqual(adjacency_matrix.shape, (3, 3))

    #     self.assertListEqual(
    #         list(adjacency_matrix.columns), ["GeneA", "GeneB", "GeneC"]
    #     )
    #     self.assertListEqual(list(adjacency_matrix.index), ["GeneA", "GeneB", "GeneC"])


if __name__ == "__main__":
    unittest.main()
