import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
pd_notebooks = project_root / "PD-Notebooks"
if str(pd_notebooks) not in sys.path:
    sys.path.insert(0, str(pd_notebooks))

from processing.parkinsons_processing import (
    log_transform_counts,
    select_hvg,
    preprocess_for_gnn,
    build_node_features,
    preprocess_pipeline,
)


class TestParkinsonsProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic expression data
        np.random.seed(42)
        n_genes = 100
        n_samples = 20

        self.expression_df = pd.DataFrame(
            np.random.poisson(lam=10, size=(n_genes, n_samples)),
            index=[f"GENE_{i}" for i in range(n_genes)],
            columns=[f"SAMPLE_{i}" for i in range(n_samples)],
        )

    def test_log_transform(self):
        """Test log transformation."""
        transformed = log_transform_counts(self.expression_df, method="log2")

        self.assertEqual(transformed.shape, self.expression_df.shape)
        self.assertTrue((transformed >= 0).all().all())  # Should be non-negative after log
        self.assertFalse(transformed.equals(self.expression_df))  # Should be different

    def test_select_hvg_variance(self):
        """Test HVG selection by variance."""
        hvg_df = select_hvg(self.expression_df, n_top=20, method="variance")

        self.assertEqual(hvg_df.shape[0], 20)
        self.assertEqual(hvg_df.shape[1], self.expression_df.shape[1])
        self.assertTrue(all(gene in self.expression_df.index for gene in hvg_df.index))

    def test_select_hvg_cv(self):
        """Test HVG selection by coefficient of variation."""
        hvg_df = select_hvg(self.expression_df, n_top=15, method="cv")

        self.assertEqual(hvg_df.shape[0], 15)
        self.assertTrue(all(gene in self.expression_df.index for gene in hvg_df.index))

    def test_preprocess_for_gnn(self):
        """Test complete preprocessing pipeline."""
        processed = preprocess_for_gnn(
            self.expression_df,
            log_transform=True,
            select_hvgs=True,
            n_hvgs=30,
            normalize=True,
        )

        self.assertIsInstance(processed, pd.DataFrame)
        self.assertLessEqual(processed.shape[0], self.expression_df.shape[0])

    def test_build_node_features_mean_variance(self):
        """Test node feature construction (mean/variance)."""
        features = build_node_features(
            self.expression_df, feature_type="mean_variance"
        )

        self.assertEqual(features.shape[0], self.expression_df.shape[0])
        self.assertEqual(features.shape[1], 2)  # mean and variance
        self.assertIn("mean", features.columns)
        self.assertIn("variance", features.columns)

    def test_build_node_features_pca(self):
        """Test node feature construction (PCA)."""
        features = build_node_features(
            self.expression_df, feature_type="pca", n_pca_components=5
        )

        self.assertEqual(features.shape[0], self.expression_df.shape[0])
        self.assertEqual(features.shape[1], 5)
        self.assertTrue(all(col.startswith("PC") for col in features.columns))

    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        processed_expr, node_features = preprocess_pipeline(
            self.expression_df,
            log_transform=True,
            select_hvgs=True,
            n_hvgs=30,
            build_features=True,
            feature_type="mean_variance",
        )

        self.assertIsInstance(processed_expr, pd.DataFrame)
        self.assertIsInstance(node_features, pd.DataFrame)
        self.assertEqual(processed_expr.shape[0], node_features.shape[0])


if __name__ == "__main__":
    unittest.main()
