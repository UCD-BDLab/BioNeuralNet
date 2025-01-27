import unittest
from unittest.mock import patch
import pandas as pd
import networkx as nx
from bioneuralnet.clustering import PageRank


class TestPageRank(unittest.TestCase):

    def setUp(self):
        """
        Set up in-memory data structures for testing.
        """
        self.G = nx.Graph()
        self.G.add_edge("1", "2", weight=1.0)
        self.G.add_edge("2", "3", weight=2.0)
        self.G.add_edge("3", "1", weight=1.5)
        self.omics_data = pd.DataFrame(
            {
                "1": [0.1, 0.2, 0.3, 0.4],
                "2": [0.3, 0.4, 0.5, 0.6],
                "3": [0.5, 0.6, 0.7, 0.8],
            },
            index=["1", "2", "3", "4"],
        )

        self.phenotype_data = pd.Series([1, 0, 1, 0], index=["1", "2", "3", "4"])

    def test_generate_weighted_personalization(self):
        """
        Test the generate_weighted_personalization method.
        """
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9,
        )

        seed_nodes = ["1", "2"]
        personalization = pagerank_instance.generate_weighted_personalization(
            seed_nodes
        )

        self.assertEqual(set(personalization.keys()), set(seed_nodes))
        for value in personalization.values():
            self.assertIsInstance(value, float)

    @patch.object(
        PageRank,
        "generate_weighted_personalization",
        return_value={"1": 0.5, "2": 0.5},
    )
    @patch.object(
        PageRank,
        "sweep_cut",
        return_value=(["1", "2"], 2, 0.1, 0.8, 0.2, "0.8 (0.05)"),
    )
    @patch.object(PageRank, "save_results")
    def test_run_pagerank_clustering(
        self, mock_save_results, mock_sweep_cut, mock_gen_pers
    ):
        """
        Test the run_pagerank_clustering method.
        """
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9,
            max_iter=100,
            tol=1e-6,
            output_dir="test_output",
        )

        with patch(
            "networkx.pagerank",
            return_value={"1": 0.6, "2": 0.4, "3": 0.0, "4": 0.0},
        ) as mock_pagerank:
            results = pagerank_instance.run_pagerank_clustering(seed_nodes=["1", "2"])

            mock_pagerank.assert_called_once_with(
                pagerank_instance.G,
                alpha=pagerank_instance.alpha,
                personalization={"1": 0.5, "2": 0.5},
                max_iter=pagerank_instance.max_iter,
                tol=pagerank_instance.tol,
                weight="weight",
            )
            mock_gen_pers.assert_called_once_with(["1", "2"])
            mock_sweep_cut.assert_called_once_with(
                {"1": 0.6, "2": 0.4, "3": 0.0, "4": 0.0}
            )
            mock_save_results.assert_called_once_with(
                {
                    "cluster_nodes": ["1", "2"],
                    "cluster_size": 2,
                    "conductance": 0.1,
                    "correlation": 0.8,
                    "composite_score": 0.2,
                    "correlation_pvalue": "0.8 (0.05)",
                }
            )

            expected_results = {
                "cluster_nodes": ["1", "2"],
                "cluster_size": 2,
                "conductance": 0.1,
                "correlation": 0.8,
                "composite_score": 0.2,
                "correlation_pvalue": "0.8 (0.05)",
            }
            self.assertEqual(results, expected_results)

    def test_run_full_pipeline(self):
        """
        Test the full run method.
        """
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9,
        )

        with patch.object(
            pagerank_instance,
            "run_pagerank_clustering",
            return_value={
                "cluster_nodes": ["1", "2"],
                "cluster_size": 2,
                "conductance": 0.1,
                "correlation": 0.8,
                "composite_score": 0.2,
                "correlation_pvalue": "0.8 (0.05)",
            },
        ) as mock_run_pagerank:

            results = pagerank_instance.run(seed_nodes=["1", "2"])
            mock_run_pagerank.assert_called_once_with(["1", "2"])
            self.assertEqual(
                results,
                {
                    "cluster_nodes": ["1", "2"],
                    "cluster_size": 2,
                    "conductance": 0.1,
                    "correlation": 0.8,
                    "composite_score": 0.2,
                    "correlation_pvalue": "0.8 (0.05)",
                },
            )

    def test_save_results(self):
        """
        Test the save_results method.
        """
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9,
        )

        results = {
            "cluster_nodes": ["1", "2"],
            "cluster_size": 2,
            "conductance": 0.1,
            "correlation": 0.8,
            "composite_score": 0.2,
            "correlation_pvalue": "0.8 (0.05)",
        }

        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv, patch(
            "os.path.join", return_value="pagerank_results_test.csv"
        ):
            pagerank_instance.save_results(results)

            mock_to_csv.assert_called_once_with(
                "pagerank_results_test.csv", index=False
            )

    def test_generate_graph_empty(self):
        """
        Test that running clustering with an empty seed_nodes list raises a ValueError.
        """
        empty_graph = nx.Graph()
        pagerank_instance = PageRank(
            graph=empty_graph,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
        )

        with self.assertRaises(ValueError) as context:
            pagerank_instance.run_pagerank_clustering(seed_nodes=[])

        self.assertEqual(str(context.exception), "Seed nodes list cannot be empty.")

    @patch.object(
        PageRank,
        "generate_weighted_personalization",
        return_value={"1": 0.5, "2": 0.5},
    )
    @patch.object(
        PageRank,
        "sweep_cut",
        return_value=(["1", "2"], 2, 0.1, 0.8, 0.2, "0.8 (0.05)"),
    )
    @patch.object(PageRank, "save_results")
    def test_run_pagerank_clustering_valid_seed_nodes(
        self, mock_save_results, mock_sweep_cut, mock_gen_pers
    ):
        """
        Test run_pagerank_clustering with valid seed nodes.
        """
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9,
            max_iter=100,
            tol=1e-6,
            output_dir="test_output",
        )

        with patch(
            "networkx.pagerank",
            return_value={"1": 0.6, "2": 0.4, "3": 0.0, "4": 0.0},
        ) as mock_pagerank:
            results = pagerank_instance.run_pagerank_clustering(seed_nodes=["1", "2"])

            mock_pagerank.assert_called_once_with(
                pagerank_instance.G,
                alpha=pagerank_instance.alpha,
                personalization={"1": 0.5, "2": 0.5},
                max_iter=pagerank_instance.max_iter,
                tol=pagerank_instance.tol,
                weight="weight",
            )
            mock_gen_pers.assert_called_once_with(["1", "2"])
            mock_sweep_cut.assert_called_once_with(
                {"1": 0.6, "2": 0.4, "3": 0.0, "4": 0.0}
            )
            mock_save_results.assert_called_once_with(
                {
                    "cluster_nodes": ["1", "2"],
                    "cluster_size": 2,
                    "conductance": 0.1,
                    "correlation": 0.8,
                    "composite_score": 0.2,
                    "correlation_pvalue": "0.8 (0.05)",
                }
            )

            expected_results = {
                "cluster_nodes": ["1", "2"],
                "cluster_size": 2,
                "conductance": 0.1,
                "correlation": 0.8,
                "composite_score": 0.2,
                "correlation_pvalue": "0.8 (0.05)",
            }
            self.assertEqual(results, expected_results)


if __name__ == "__main__":
    unittest.main()
