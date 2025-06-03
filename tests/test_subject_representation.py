import unittest
import pandas as pd
from bioneuralnet.downstream_task import SubjectRepresentation

class TestSubjectRepresentation(unittest.TestCase):

    def setUp(self):
        self.omics_data = pd.DataFrame({"gene1": [1, 2, 3],"gene2": [4, 5, 6],"gene3": [7, 8, 9],},index=["sample1", "sample2", "sample3"],)
        self.phenotype_data = pd.DataFrame({"phenotype": [0, 1, 2]}, index=["sample1", "sample2", "sample3"])
        self.precomputed_embeddings = pd.DataFrame({"dim1": [0.1, 0.2, 0.3],"dim2": [0.4, 0.5, 0.6],"dim3": [0.7, 0.8, 0.9],},index=["gene1", "gene2", "gene3"],)

    def test_run_with_precomputed_embeddings(self):
        sr = SubjectRepresentation(
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            embeddings=self.precomputed_embeddings,
            reduce_method="PCA",
            tune=False,
        )

        enhanced = sr.run()
        self.assertIsInstance(enhanced, pd.DataFrame)
        self.assertEqual(enhanced.shape[0], 3)
        self.assertEqual(set(enhanced.columns), {"gene1", "gene2", "gene3"})

    def test_init_without_embeddings(self):
        # Case A: embeddings=None
        with self.assertRaises(ValueError):
            SubjectRepresentation(
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                embeddings=None,
                reduce_method="PCA",
                tune=False,
            )

        with self.assertRaises(ValueError):
            SubjectRepresentation(
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                embeddings=pd.DataFrame(),
                reduce_method="PCA",
                tune=False,
            )

    def test_init_with_wrong_embedding_type(self):
        bad_embeddings = ["gene1", "gene2", "gene3"]
        with self.assertRaises(ValueError):
            SubjectRepresentation(
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                embeddings=bad_embeddings,
                reduce_method="PCA",
                tune=False,
            )

    def test_integrate_embeddings_wrong_type(self):
        sr = SubjectRepresentation(
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            embeddings=self.precomputed_embeddings,
            reduce_method="PCA",
            tune=False,
        )

        with self.assertRaises(ValueError):
            sr._integrate_embeddings(reduced=123, method="multiply", alpha=1.0, beta=0.5)

if __name__ == "__main__":
    unittest.main()
