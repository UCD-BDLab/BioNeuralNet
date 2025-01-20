import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from bioneuralnet.external_tools import WGCNA
import os
import subprocess


class TestWGCNA(unittest.TestCase):

    def setUp(self):
        self.phenotype_df = pd.DataFrame(
            {
                "SampleID": ["S1", "S2", "S3", "S4"],
                "Phenotype": ["Control", "Treatment", "Control", "Treatment"],
            }
        )

        self.omics_df1 = pd.DataFrame(
            {
                "SampleID": ["S1", "S2", "S3", "S4"],
                "GeneA": [1.2, 2.3, 3.1, 4.0],
                "GeneB": [2.1, 3.4, 1.2, 3.3],
                "GeneC": [3.3, 1.5, 2.2, 4.1],
            }
        )

        self.omics_df2 = pd.DataFrame(
            {
                "SampleID": ["S1", "S2", "S3", "S4"],
                "GeneD": [4.2, 5.3, 6.1, 7.0],
                "GeneE": [5.1, 6.4, 4.2, 6.3],
                "GeneF": [6.3, 4.5, 5.2, 7.1],
            }
        )

        self.omics_dfs = [self.omics_df1, self.omics_df2]
        self.data_types = ["Transcriptomics", "Proteomics"]

    @patch("bioneuralnet.external_tools.wgcna.subprocess.run")
    def test_wgcna_successful_run(self, mock_run):
        mock_completed_process = MagicMock()
        mock_completed_process.returncode = 0
        mock_completed_process.stdout = '{"columns":["GeneA","GeneB","GeneC","GeneD","GeneE","GeneF"],"index":["GeneA","GeneB","GeneC","GeneD","GeneE","GeneF"],"data":[[1,0.8,0.3,0.5,0.2,0.1],[0.8,1,0.4,0.6,0.3,0.2],[0.3,0.4,1,0.7,0.4,0.3],[0.5,0.6,0.7,1,0.5,0.4],[0.2,0.3,0.4,0.5,1,0.5],[0.1,0.2,0.3,0.4,0.5,1]]}'
        mock_run.return_value = mock_completed_process

        wgcna = WGCNA(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
        )
        adjacency_matrix = wgcna.run()
        self.assertIsInstance(adjacency_matrix, pd.DataFrame)
        self.assertFalse(adjacency_matrix.isnull().values.any())
        self.assertEqual(adjacency_matrix.shape, (6, 6))
        self.assertListEqual(
            list(adjacency_matrix.columns),
            ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE", "GeneF"],
        )
        self.assertListEqual(
            list(adjacency_matrix.index),
            ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE", "GeneF"],
        )
        self.assertAlmostEqual(adjacency_matrix.loc["GeneA", "GeneB"], 0.8)
        self.assertAlmostEqual(adjacency_matrix.loc["GeneD", "GeneF"], 0.4)

    @patch("bioneuralnet.external_tools.wgcna.subprocess.run")
    def test_wgcna_run_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="Rscript WGCNA.R",
            stderr="Error in readLines(con = \"stdin\") : incomplete final line found on 'stdin'\n",
        )

        wgcna = WGCNA(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
        )

        with self.assertRaises(subprocess.CalledProcessError):
            wgcna.run()

    def test_mismatched_omics_and_data_types(self):
        with self.assertRaises(ValueError):
            WGCNA(
                phenotype_df=self.phenotype_df,
                omics_dfs=self.omics_dfs,
                data_types=["Transcriptomics"],
                soft_power=6,
                min_module_size=30,
                merge_cut_height=0.25,
            )

    @patch("bioneuralnet.external_tools.wgcna.subprocess.run")
    def test_no_valid_samples(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="Rscript WGCNA.R",
            stderr="Error: No valid samples after preprocessing.\nExecution halted",
        )

        self.omics_dfs[0].iloc[0, 1] = pd.NA
        self.omics_dfs[1].iloc[1, 2] = pd.NA

        wgcna = WGCNA(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
        )

        with self.assertRaises(subprocess.CalledProcessError):
            wgcna.run()

    @patch("bioneuralnet.external_tools.wgcna.subprocess.run")
    def test_save_adjacency_matrix(self, mock_run):
        mock_completed_process = MagicMock()
        mock_completed_process.returncode = 0
        mock_completed_process.stdout = '{"columns":["GeneA","GeneB","GeneC","GeneD","GeneE","GeneF"],"index":["GeneA","GeneB","GeneC","GeneD","GeneE","GeneF"],"data":[[1,0.8,0.3,0.5,0.2,0.1],[0.8,1,0.4,0.6,0.3,0.2],[0.3,0.4,1,0.7,0.4,0.3],[0.5,0.6,0.7,1,0.5,0.4],[0.2,0.3,0.4,0.5,1,0.5],[0.1,0.2,0.3,0.4,0.5,1]]}'
        mock_run.return_value = mock_completed_process

        wgcna = WGCNA(
            phenotype_df=self.phenotype_df,
            omics_dfs=self.omics_dfs,
            data_types=self.data_types,
            soft_power=6,
            min_module_size=30,
            merge_cut_height=0.25,
        )
        adjacency_matrix = wgcna.run()
        save_path = "test_adjacency_matrix.json"
        adjacency_matrix.to_json(save_path, orient="split")
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
