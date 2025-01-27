import unittest
import pandas as pd

from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.downstream_task.dpmon import DPMON


class TestDPMONRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load sample dataset and generate adjacency with SmCCNet
        """
        cls.datasets = DatasetLoader()
        cls.omics1, cls.omics2, cls.pheno = cls.datasets("example1")

        cls.omics_list = [cls.omics1, cls.omics2]
        cls.phenotype_data = cls.pheno.rename(columns={"Pheno": "phenotype"})

        cls.smccnet = SmCCNet(
            phenotype_df=cls.phenotype_data,
            omics_dfs=cls.omics_list,
            data_types=["Genomics", "miRNA"],
            kfold=5,
            summarization="PCA",
            seed=127,
        )
        cls.adjacency_matrix = cls.smccnet.run()

    def test_dpmon_run(self):
        """
        Test DPMON on real adjacency + real multi-omics data
        """

        dpmon = DPMON(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=self.omics_list,
            phenotype_data=self.phenotype_data,
            clinical_data=None,
            model="GAT",
            gnn_hidden_dim=16,
            layer_num=2,
            nn_hidden_dim1=8,
            nn_hidden_dim2=8,
            epoch_num=1,
            repeat_num=1,
            lr=0.01,
            weight_decay=1e-4,
            gpu=False,
            tune=False,
        )

        predictions = dpmon.run()

        self.assertIsInstance(predictions, pd.DataFrame)
        if not predictions.empty:
            self.assertIn("Actual", predictions.columns)
            self.assertIn("Predicted", predictions.columns)


if __name__ == "__main__":
    unittest.main()
