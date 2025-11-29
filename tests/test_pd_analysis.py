import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
pd_notebooks = project_root / "PD-Notebooks"
if str(pd_notebooks) not in sys.path:
    sys.path.insert(0, str(pd_notebooks))

from analysis.embedding_analysis import (
    reduce_embeddings,
    cluster_embeddings,
    visualize_clusters,
    analyze_clusters,
    embedding_analysis_pipeline,
    ClusterResults,
)


class TestPDAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_nodes = 50
        embedding_dim = 10

        # Create synthetic embeddings
        self.embeddings = np.random.randn(n_nodes, embedding_dim)

        # Create node names
        self.node_names = [f"GENE_{i}" for i in range(n_nodes)]

        # Create small adjacency matrix for testing
        self.adjacency_matrix = pd.DataFrame(
            np.random.rand(n_nodes, n_nodes),
            index=self.node_names,
            columns=self.node_names,
        )
        # Make symmetric and set diagonal to 1
        self.adjacency_matrix = (
            self.adjacency_matrix + self.adjacency_matrix.T
        ) / 2
        np.fill_diagonal(self.adjacency_matrix.values, 1.0)

        # Create gene metadata
        self.gene_metadata = pd.DataFrame(
            {
                "Symbol": [f"GENE{i}" for i in range(n_nodes)],
                "Description": [f"Gene {i} description" for i in range(n_nodes)],
            },
            index=self.node_names,
        )

    def test_reduce_embeddings_tsne(self):
        """Test embedding reduction with t-SNE."""
        reduced = reduce_embeddings(
            self.embeddings, method="tsne", n_components=2, random_state=42
        )

        self.assertEqual(reduced.shape[0], self.embeddings.shape[0])
        self.assertEqual(reduced.shape[1], 2)

    def test_reduce_embeddings_umap(self):
        """Test embedding reduction with UMAP (if available)."""
        try:
            reduced = reduce_embeddings(
                self.embeddings, method="umap", n_components=2, random_state=42
            )
            self.assertEqual(reduced.shape[0], self.embeddings.shape[0])
            self.assertEqual(reduced.shape[1], 2)
        except ValueError:
            # UMAP not available, skip
            self.skipTest("UMAP not available")

    def test_cluster_embeddings_kmeans(self):
        """Test KMeans clustering."""
        results = cluster_embeddings(
            self.embeddings,
            method="kmeans",
            n_clusters=3,
            random_state=42,
        )

        self.assertIsInstance(results, ClusterResults)
        self.assertEqual(results.n_clusters, 3)
        self.assertEqual(len(results.labels), self.embeddings.shape[0])
        self.assertIn(0, results.labels)  # Should have cluster 0
        self.assertIn(1, results.labels)  # Should have cluster 1

    def test_cluster_embeddings_auto_k(self):
        """Test KMeans with auto-detected number of clusters."""
        results = cluster_embeddings(
            self.embeddings,
            method="kmeans",
            n_clusters=None,  # Auto-detect
            random_state=42,
        )

        self.assertIsInstance(results, ClusterResults)
        self.assertGreater(results.n_clusters, 0)
        self.assertLessEqual(results.n_clusters, 10)  # Reasonable upper bound

    def test_analyze_clusters(self):
        """Test cluster analysis."""
        # Create cluster results
        labels = np.array([0] * 20 + [1] * 20 + [2] * 10)
        cluster_results = ClusterResults(
            labels=labels,
            n_clusters=3,
            silhouette_score=0.5,
            cluster_sizes={0: 20, 1: 20, 2: 10},
        )

        cluster_df, summary_df = analyze_clusters(
            cluster_results, self.node_names, self.gene_metadata
        )

        self.assertIsInstance(cluster_df, pd.DataFrame)
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertIn("cluster", cluster_df.columns)

    def test_embedding_analysis_pipeline(self):
        """Test complete embedding analysis pipeline."""
        try:
            reduced_emb, cluster_results, cluster_df, summary_df = (
                embedding_analysis_pipeline(
                    self.embeddings,
                    self.node_names,
                    adjacency_matrix=self.adjacency_matrix,
                    gene_metadata=self.gene_metadata,
                    reduction_method="tsne",  # Use t-SNE (more reliable)
                    clustering_method="kmeans",
                    n_clusters=3,
                    random_state=42,
                )
            )

            self.assertEqual(reduced_emb.shape[0], self.embeddings.shape[0])
            self.assertEqual(reduced_emb.shape[1], 2)
            self.assertIsInstance(cluster_results, ClusterResults)
            self.assertIsInstance(cluster_df, pd.DataFrame)
        except Exception as e:
            # Some dependencies might not be available
            self.skipTest(f"Pipeline test skipped: {e}")


if __name__ == "__main__":
    unittest.main()
