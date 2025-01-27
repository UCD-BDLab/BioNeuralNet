import unittest
import torch
import pandas as pd

from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.subject_representation import GraphEmbedding


class TestSubjectRepresentationRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load sample dataset via DatasetLoader and generate adjacency with SmCCNet
        """
        cls.datasets = DatasetLoader()
        cls.omics1, cls.omics2, cls.pheno = cls.datasets("example1")
        cls.omics = pd.concat([cls.omics1, cls.omics2], axis=1)
        cls.pheno_data = cls.pheno.rename(columns={"Pheno": "phenotype"})

        cls.smccnet = SmCCNet(
            phenotype_df=cls.pheno_data,
            omics_dfs=[cls.omics],
            data_types=["Genomics+miRNA"],
            kfold=5,
            summarization="PCA",
            seed=127,
        )
        cls.adjacency_matrix = cls.smccnet.run()

    def test_graph_embedding_pipeline(self):
        """
        Test GraphEmbedding with real adjacency + real omics data (GNN-based approach).
        """
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics,
            phenotype_data=self.pheno_data,
            clinical_data=None,
            embeddings=None,
            reduce_method="PCA",
        )

        enhanced_omics_data = graph_embed.run()

        self.assertIsInstance(enhanced_omics_data, pd.DataFrame)
        self.assertFalse(
            enhanced_omics_data.empty, "Enhanced omics data should not be empty."
        )

        self.assertEqual(
            enhanced_omics_data.shape[0],
            self.omics.shape[0],
            "Enhanced data must have same #samples (rows) as original omics.",
        )


if __name__ == "__main__":
    unittest.main()
