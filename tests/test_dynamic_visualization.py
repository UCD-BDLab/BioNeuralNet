import pytest
import pandas as pd
import networkx as nx
from bioneuralnet.external_tools import DynamicVisualizer
from unittest import mock


@pytest.fixture
def sample_adjacency_matrix(tmp_path):
    """
    Creates a sample adjacency matrix for testing.
    """
    data = {"gene1": [0, 1, 0], "gene2": [1, 0, 1], "gene3": [0, 1, 0]}
    adjacency_matrix = pd.DataFrame(data, index=["gene1", "gene2", "gene3"])
    return adjacency_matrix


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Provides a temporary directory for output files.
    """
    return tmp_path / "output"


def test_generate_graph(sample_adjacency_matrix):
    """
    Test that DynamicVisualizer generates the correct NetworkX graph from the adjacency matrix.
    """
    visualizer = DynamicVisualizer(adjacency_matrix=sample_adjacency_matrix)
    G = visualizer.generate_graph()

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert "gene1" in G
    assert "gene2" in G
    assert "gene3" in G
    assert G.has_edge("gene1", "gene2")
    assert G.has_edge("gene2", "gene3")
    assert not G.has_edge("gene1", "gene3")


def test_visualize_creates_file(sample_adjacency_matrix, temp_output_dir):
    """
    Test that the visualize method creates the expected HTML file.
    """
    visualizer = DynamicVisualizer(
        adjacency_matrix=sample_adjacency_matrix,
        output_dir=temp_output_dir,
        output_filename="test_dynamic_network.html",
    )

    with mock.patch(
        "bioneuralnet.external_tools.dynamic_visualization.get_logger"
    ) as mock_get_logger:
        mock_logger = mock.Mock()
        mock_get_logger.return_value = mock_logger
        G = visualizer.generate_graph()
        visualizer.visualize(G)

    output_file = temp_output_dir / "test_dynamic_network.html"
    assert output_file.exists()
    assert output_file.is_file()


def test_visualize_with_invalid_layout(sample_adjacency_matrix, temp_output_dir):
    """
    Test that the visualize method handles invalid layout options gracefully.
    """
    invalid_layout = "invalid_layout"
    visualizer = DynamicVisualizer(
        adjacency_matrix=sample_adjacency_matrix,
        layout=invalid_layout,
        output_dir=temp_output_dir,
        output_filename="dynamic_network_invalid.html",
    )

    with mock.patch.object(visualizer.logger, "warning") as mock_warning:
        G = visualizer.generate_graph()
        visualizer.visualize(G)

        output_file = temp_output_dir / "dynamic_network_invalid.html"
        assert output_file.exists()
        assert output_file.is_file()

        mock_warning.assert_called_with(
            f"Layout '{invalid_layout}' not recognized. Falling back to spring layout."
        )


def test_visualize_with_empty_adjacency_matrix(tmp_path):
    """
    Test that the visualize method handles an empty adjacency matrix.
    """
    empty_adjacency = pd.DataFrame()
    visualizer = DynamicVisualizer(
        adjacency_matrix=empty_adjacency,
        output_dir=tmp_path,
        output_filename="dynamic_network_empty.html",
    )

    with pytest.raises(nx.NetworkXError):
        G = visualizer.generate_graph()
        visualizer.visualize(G)


def test_visualize_output_directory_creation(sample_adjacency_matrix, tmp_path):
    """
    Test that the visualize method creates the output directory if it does not exist.
    """
    non_existent_dir = tmp_path / "non_existent_output"
    visualizer = DynamicVisualizer(
        adjacency_matrix=sample_adjacency_matrix,
        output_dir=non_existent_dir,
        output_filename="dynamic_network_new_dir.html",
    )

    with mock.patch(
        "bioneuralnet.external_tools.dynamic_visualization.get_logger"
    ) as mock_get_logger:
        mock_logger = mock.Mock()
        mock_get_logger.return_value = mock_logger
        G = visualizer.generate_graph()
        visualizer.visualize(G)

    output_file = non_existent_dir / "dynamic_network_new_dir.html"
    assert output_file.exists()
    assert output_file.is_file()


def test_visualize_with_custom_visualization_params(
    sample_adjacency_matrix, temp_output_dir
):
    """
    Test that custom visualization parameters are applied correctly.
    """
    visualizer = DynamicVisualizer(
        adjacency_matrix=sample_adjacency_matrix,
        layout="hierarchical",
        bgcolor="#f0f0f0",
        font_color="blue",
        output_dir=temp_output_dir,
        output_filename="dynamic_network_custom.html",
        width="80%",
        height="600px",
    )

    with mock.patch(
        "bioneuralnet.external_tools.dynamic_visualization.get_logger"
    ) as mock_get_logger:
        mock_logger = mock.Mock()
        mock_get_logger.return_value = mock_logger
        G = visualizer.generate_graph()
        visualizer.visualize(G)

    output_file = temp_output_dir / "dynamic_network_custom.html"
    assert output_file.exists()
    assert output_file.is_file()
