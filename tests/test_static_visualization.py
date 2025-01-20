# import pytest
# import pandas as pd
# import networkx as nx

# from bioneuralnet.external_tools import StaticVisualizer
# from unittest import mock


# @pytest.fixture
# def sample_adjacency_matrix(tmp_path):
#     """
#     Creates a sample adjacency matrix for testing.
#     """
#     data = {"gene1": [0, 1, 0], "gene2": [1, 0, 1], "gene3": [0, 1, 0]}
#     adjacency_matrix = pd.DataFrame(data, index=["gene1", "gene2", "gene3"])
#     return adjacency_matrix


# @pytest.fixture
# def mock_logger():
#     with mock.patch("bioneuralnet.utils.logger.get_logger") as mock_get_logger:
#         mock_logger = mock.Mock()
#         mock_get_logger.return_value = mock_logger
#         yield mock_logger


# @pytest.fixture
# def temp_output_dir(tmp_path):
#     """
#     Provides a temporary directory for output files.
#     """
#     return tmp_path / "output"


# def test_generate_graph(sample_adjacency_matrix):
#     """
#     Test that StaticVisualizer generates the correct NetworkX graph from the adjacency matrix.
#     """
#     visualizer = StaticVisualizer(adjacency_matrix=sample_adjacency_matrix)
#     G = visualizer.generate_graph()

#     assert isinstance(G, nx.Graph)
#     assert G.number_of_nodes() == 3
#     assert G.number_of_edges() == 2
#     assert "gene1" in G
#     assert "gene2" in G
#     assert "gene3" in G
#     assert G.has_edge("gene1", "gene2")
#     assert G.has_edge("gene2", "gene3")
#     assert not G.has_edge("gene1", "gene3")


# def test_visualize_creates_file(sample_adjacency_matrix, temp_output_dir):
#     """
#     Test that the visualize method creates the expected image file.
#     """
#     visualizer = StaticVisualizer(
#         adjacency_matrix=sample_adjacency_matrix,
#         output_dir=temp_output_dir,
#         output_filename="test_static_network.png",
#     )

#     with mock.patch("bioneuralnet.utils.logger.get_logger") as mock_get_logger:
#         mock_logger = mock.Mock()
#         mock_get_logger.return_value = mock_logger
#         G = visualizer.generate_graph()
#         visualizer.visualize(G)

#     output_file = temp_output_dir / "test_static_network.png"
#     assert output_file.exists()
#     assert output_file.is_file()


# def test_visualize_with_empty_adjacency_matrix(tmp_path):
#     """
#     Test that the visualize method handles an empty adjacency matrix.
#     """
#     empty_adjacency = pd.DataFrame()
#     visualizer = StaticVisualizer(
#         adjacency_matrix=empty_adjacency,
#         output_dir=tmp_path,
#         output_filename="static_network_empty.png",
#     )

#     with mock.patch("bioneuralnet.utils.logger.get_logger") as mock_get_logger:
#         mock_logger = mock.Mock()
#         mock_get_logger.return_value = mock_logger
#         with pytest.raises(nx.NetworkXError):
#             G = visualizer.generate_graph()
#             visualizer.visualize(G)


# def test_visualize_output_directory_creation(sample_adjacency_matrix, tmp_path):
#     """
#     Test that the visualize method creates the output directory if it does not exist.
#     """
#     non_existent_dir = tmp_path / "non_existent_output"

#     with mock.patch("bioneuralnet.utils.logger.get_logger") as mock_get_logger:
#         mock_logger = mock.Mock()
#         mock_get_logger.return_value = mock_logger

#         visualizer = StaticVisualizer(
#             adjacency_matrix=sample_adjacency_matrix,
#             output_dir=non_existent_dir,
#             output_filename="static_network_new_dir.png",
#         )

#         G = visualizer.generate_graph()
#         visualizer.visualize(G)

#     output_file = non_existent_dir / "static_network_new_dir.png"
#     assert output_file.exists()
#     assert output_file.is_file()


# def test_visualize_with_custom_visualization_params(
#     sample_adjacency_matrix, temp_output_dir
# ):
#     """
#     Test that custom visualization parameters are applied correctly.
#     """
#     visualizer = StaticVisualizer(
#         adjacency_matrix=sample_adjacency_matrix,
#         layout="circular",
#         node_size=500,
#         node_color="red",
#         edge_color="blue",
#         linewidths=2.0,
#         font_size=12,
#         output_dir=temp_output_dir,
#         output_filename="static_network_custom.png",
#     )

#     with mock.patch("bioneuralnet.utils.logger.get_logger") as mock_get_logger:
#         mock_logger = mock.Mock()
#         mock_get_logger.return_value = mock_logger
#         G = visualizer.generate_graph()
#         visualizer.visualize(G)

#     output_file = temp_output_dir / "static_network_custom.png"
#     assert output_file.exists()
#     assert output_file.is_file()
