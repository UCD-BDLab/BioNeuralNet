import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import networkx as nx
from bioneuralnet.clustering import PageRank


class TestPageRank(unittest.TestCase):

    def setUp(self):
        """
        Set up in-memory data structures for testing.
        """
        # Create a simple graph
        self.G = nx.Graph()
        self.G.add_edge('1', '2', weight=1.0)
        self.G.add_edge('2', '3', weight=2.0)
        self.G.add_edge('3', '4', weight=1.5)

        # Create omics data
        self.omics_data = pd.DataFrame({
            'Gene1': [0.1, 0.2, 0.3, 0.4],
            'Gene2': [0.3, 0.4, 0.5, 0.6],
            'Gene3': [0.5, 0.6, 0.7, 0.8]
        }, index=['1', '2', '3', '4'])

        # Create phenotype data
        self.phenotype_data = pd.Series([1, 0, 1, 0], index=['1', '2', '3', '4'])

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_generate_weighted_personalization(self, mock_get_logger):
        """
        Test the generate_weighted_personalization method.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9
        )

        seed_nodes = ['1', '2']
        personalization = pagerank_instance.generate_weighted_personalization(seed_nodes)

        # Since generate_weighted_personalization may return any float values,
        # we primarily check the keys and that the values are floats.
        self.assertEqual(set(personalization.keys()), set(seed_nodes))
        for value in personalization.values():
            self.assertIsInstance(value, float)

        # No assertion on logger.info since it's not called in generate_weighted_personalization

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_run_pagerank_clustering(self, mock_get_logger):
        """
        Test the run_pagerank_clustering method.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Initialize with in-memory data
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9
        )

        # Mock the methods called within run_pagerank_clustering
        with patch.object(pagerank_instance, 'generate_weighted_personalization', return_value={'1': 0.5, '2': 0.5}) as mock_gen_pers, \
             patch.object(pagerank_instance, 'sweep_cut', return_value=(['1', '2'], 2, 0.1, 0.8, 0.2, '0.8 (0.05)')) as mock_sweep_cut, \
             patch.object(pagerank_instance, 'save_results') as mock_save_results:

            # Mock the pagerank function
            with patch('networkx.pagerank', return_value={'1': 0.6, '2': 0.4, '3': 0.0, '4': 0.0}) as mock_pagerank:
                results = pagerank_instance.run_pagerank_clustering(seed_nodes=['1', '2'])

                # Assertions
                mock_pagerank.assert_called_once_with(
                    pagerank_instance.G,
                    alpha=pagerank_instance.alpha,
                    personalization={'1': 0.5, '2': 0.5},
                    max_iter=pagerank_instance.max_iter,
                    tol=pagerank_instance.tol,
                    weight='weight'
                )
                mock_gen_pers.assert_called_once_with(['1', '2'])
                mock_sweep_cut.assert_called_once_with({'1': 0.6, '2': 0.4, '3': 0.0, '4': 0.0})
                mock_save_results.assert_called_once_with({
                    'cluster_nodes': ['1', '2'],
                    'cluster_size': 2,
                    'conductance': 0.1,
                    'correlation': 0.8,
                    'composite_score': 0.2,
                    'correlation_pvalue': '0.8 (0.05)'
                })

                # Check the results
                expected_results = {
                    'cluster_nodes': ['1', '2'],
                    'cluster_size': 2,
                    'conductance': 0.1,
                    'correlation': 0.8,
                    'composite_score': 0.2,
                    'correlation_pvalue': '0.8 (0.05)'
                }
                self.assertEqual(results, expected_results)

                # Optionally, verify logging
                mock_logger.info.assert_any_call("Generated personalization vector for seed nodes: ['1', '2']")
                mock_logger.info.assert_any_call("PageRank computation completed.")
                mock_logger.info.assert_any_call(f"Sweep cut resulted in cluster of size {2} with conductance {0.1} and correlation {0.8}.")

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_run_full_pipeline(self, mock_get_logger):
        """
        Test the full run method.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9
        )

        # Mock the methods called within run
        with patch.object(pagerank_instance, 'run_pagerank_clustering', return_value={
            'cluster_nodes': ['1', '2'],
            'cluster_size': 2,
            'conductance': 0.1,
            'correlation': 0.8,
            'composite_score': 0.2,
            'correlation_pvalue': '0.8 (0.05)'
        }) as mock_run_pagerank:

            results = pagerank_instance.run(seed_nodes=['1', '2'])

            # Assertions
            mock_run_pagerank.assert_called_once_with(['1', '2'])
            self.assertEqual(results, {
                'cluster_nodes': ['1', '2'],
                'cluster_size': 2,
                'conductance': 0.1,
                'correlation': 0.8,
                'composite_score': 0.2,
                'correlation_pvalue': '0.8 (0.05)'
            })

            # Optionally, verify logging
            mock_logger.info.assert_any_call("PageRank clustering completed successfully.")

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_save_results(self, mock_get_logger):
        """
        Test the save_results method.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9
        )

        # Sample results
        results = {
            'cluster_nodes': ['1', '2'],
            'cluster_size': 2,
            'conductance': 0.1,
            'correlation': 0.8,
            'composite_score': 0.2,
            'correlation_pvalue': '0.8 (0.05)'
        }

        # Mock pandas to_csv method and os.path.join
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            with patch('os.path.join', return_value='pagerank_results_test.csv'):
                pagerank_instance.save_results(results)

                # Assertions
                mock_to_csv.assert_called_once_with('pagerank_results_test.csv', index=False)
                mock_logger.info.assert_any_call("Clustering results saved to pagerank_results_test.csv")

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_generate_graph_empty(self, mock_get_logger):
        """
        Test that running clustering with an empty seed_nodes list raises a ValueError.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        empty_graph = nx.Graph()
        pagerank_instance = PageRank(
            graph=empty_graph,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data
        )

        with self.assertRaises(ValueError) as context:
            # Attempting to run clustering with empty seed nodes
            pagerank_instance.run_pagerank_clustering(seed_nodes=[])

        self.assertEqual(str(context.exception), "Seed nodes list cannot be empty.")

        # Ensure that an error was logged
        mock_logger.error.assert_called_with("No seed nodes provided for PageRank clustering.")

    # @patch('bioneuralnet.clustering.pagerank.get_logger')
    # def test_run_pagerank_clustering_nonexistent_seed_nodes(self, mock_get_logger):
    #     """
    #     Test run_pagerank_clustering with seed nodes not present in the graph.
    #     """
    #     # Create a mock logger
    #     mock_logger = MagicMock()
    #     mock_get_logger.return_value = mock_logger

    #     pagerank_instance = PageRank(
    #         graph=self.G,
    #         omics_data=self.omics_data,
    #         phenotype_data=self.phenotype_data,
    #         alpha=0.9,
    #         k=0.9
    #     )

    #     seed_nodes = ['5', '6']  # Nodes not present in the graph

    #     with self.assertRaises(ValueError) as context:
    #         pagerank_instance.run_pagerank_clustering(seed_nodes=seed_nodes)

    #     self.assertEqual(str(context.exception), "Seed nodes not in graph: {'5', '6'}")

    #     # Ensure that an error was logged
    #     mock_logger.error.assert_called_with("Seed nodes not in graph: {'5', '6'}")

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_generate_weighted_personalization_single_node(self, mock_get_logger):
        """
        Test generate_weighted_personalization with a single seed node.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9
        )

        seed_nodes = ['1']
        personalization = pagerank_instance.generate_weighted_personalization(seed_nodes)

        # Since there's only one seed node, personalization should have one entry
        self.assertEqual(set(personalization.keys()), set(seed_nodes))
        self.assertEqual(len(personalization), 1)
        self.assertIsInstance(list(personalization.values())[0], float)

        # Check that the logger warned about insufficient data
        mock_logger.warning.assert_called_with("Not enough nodes (1) for correlation. Returning 0 correlation.")

    @patch('bioneuralnet.clustering.pagerank.get_logger')
    def test_run_pagerank_clustering_valid_seed_nodes(self, mock_get_logger):
        """
        Test run_pagerank_clustering with valid seed nodes.
        """
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Initialize with in-memory data
        pagerank_instance = PageRank(
            graph=self.G,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            alpha=0.9,
            k=0.9
        )

        # Mock the methods called within run_pagerank_clustering
        with patch.object(pagerank_instance, 'generate_weighted_personalization', return_value={'1': 0.5, '2': 0.5}) as mock_gen_pers, \
             patch.object(pagerank_instance, 'sweep_cut', return_value=(['1', '2'], 2, 0.1, 0.8, 0.2, '0.8 (0.05)')) as mock_sweep_cut, \
             patch.object(pagerank_instance, 'save_results') as mock_save_results:

            # Mock the pagerank function
            with patch('networkx.pagerank', return_value={'1': 0.6, '2': 0.4, '3': 0.0, '4': 0.0}) as mock_pagerank:
                results = pagerank_instance.run_pagerank_clustering(seed_nodes=['1', '2'])

                # Assertions
                mock_pagerank.assert_called_once_with(
                    pagerank_instance.G,
                    alpha=pagerank_instance.alpha,
                    personalization={'1': 0.5, '2': 0.5},
                    max_iter=pagerank_instance.max_iter,
                    tol=pagerank_instance.tol,
                    weight='weight'
                )
                mock_gen_pers.assert_called_once_with(['1', '2'])
                mock_sweep_cut.assert_called_once_with({'1': 0.6, '2': 0.4, '3': 0.0, '4': 0.0})
                mock_save_results.assert_called_once_with({
                    'cluster_nodes': ['1', '2'],
                    'cluster_size': 2,
                    'conductance': 0.1,
                    'correlation': 0.8,
                    'composite_score': 0.2,
                    'correlation_pvalue': '0.8 (0.05)'
                })

                # Check the results
                expected_results = {
                    'cluster_nodes': ['1', '2'],
                    'cluster_size': 2,
                    'conductance': 0.1,
                    'correlation': 0.8,
                    'composite_score': 0.2,
                    'correlation_pvalue': '0.8 (0.05)'
                }
                self.assertEqual(results, expected_results)

                # Optionally, verify logging
                mock_logger.info.assert_any_call("Generated personalization vector for seed nodes: ['1', '2']")
                mock_logger.info.assert_any_call("PageRank computation completed.")
                mock_logger.info.assert_any_call(f"Sweep cut resulted in cluster of size {2} with conductance {0.1} and correlation {0.8}.")

    if __name__ == '__main__':
        unittest.main()
