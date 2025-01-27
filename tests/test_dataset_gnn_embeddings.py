import unittest
import torch
import pandas as pd

from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.network_embedding import GNNEmbedding


class TestGNNEmbeddingRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load sample dataset via DatasetLoader and generate adjacency with SmCCNet
        """
        cls.datasets = DatasetLoader()
        cls.omics1, cls.omics2, cls.pheno = cls.datasets("example1")

        cls.smccnet = SmCCNet(
            phenotype_df=cls.pheno,
            omics_dfs=[cls.omics1, cls.omics2],
            data_types=["Genomics", "miRNA"],
            kfold=5,
            summarization="PCA",
            seed=127,
        )
        cls.adjacency_matrix = cls.smccnet.run()
        cls.omics = pd.concat([cls.omics1, cls.omics2], axis=1)

        cls.pheno_data = cls.pheno.rename(columns={"Pheno": "phenotype"})

    def test_gnn_embedding_run(self):
        """
        Test GNNEmbedding with real adjacency + multi-omics data to produce node embeddings.
        """
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics,
            phenotype_data=self.pheno_data,
            clinical_data=None,
            phenotype_col="phenotype",
            model_type="GAT",
            hidden_dim=32,
            layer_num=2,
            dropout=True,
            num_epochs=1,
            lr=1e-3,
            weight_decay=1e-4,
            gpu=False,
        )

        gnn.fit()
        node_embeddings = gnn.embed()

        self.assertIsInstance(
            node_embeddings, torch.Tensor, "Embeddings must be a torch.Tensor"
        )
        self.assertEqual(
            node_embeddings.shape[0],
            self.adjacency_matrix.shape[0],
            "Number of node embeddings must match adjacency dimension.",
        )
        self.assertGreater(
            node_embeddings.shape[1],
            1,
            "Embeddings dimension should be >1 (multi-dimensional).",
        )


if __name__ == "__main__":
    unittest.main()
