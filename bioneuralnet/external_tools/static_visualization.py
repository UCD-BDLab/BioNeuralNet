import os
from typing import Optional
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ..utils.logger import get_logger


class StaticVisualizer:
    """
    StaticVisualizer Class for Generating Static Network Visualizations.

    Utilizes NetworkX and Matplotlib to create and save static images of networks.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        layout: str = "spring",
        node_size: int = 300,
        node_color: str = "skyblue",
        edge_color: str = "gray",
        linewidths: float = 1.0,
        font_size: int = 10,
        output_dir: Optional[str] = None,
        output_filename: str = "static_network.png",
    ):
        """
        Initializes the StaticVisualizer instance.

        Args:
            adjacency_matrix (pd.DataFrame): Adjacency matrix representing the network.
            layout (str, optional): Layout algorithm for network visualization ('spring', 'kamada_kawai', 'circular', etc.). Defaults to 'spring'.
            node_size (int, optional): Size of the nodes in the visualization. Defaults to 300.
            node_color (str, optional): Color of the nodes. Defaults to 'skyblue'.
            edge_color (str, optional): Color of the edges. Defaults to 'gray'.
            linewidths (float, optional): Width of the edges. Defaults to 1.0.
            font_size (int, optional): Font size for node labels. Defaults to 10.
            output_dir (str, optional): Directory to save the visualization image. If None, saves in the current directory. Defaults to None.
            output_filename (str, optional): Filename for the saved visualization image. Defaults to "static_network.png".
        """
        self.adjacency_matrix = adjacency_matrix
        self.layout = layout
        self.node_size = node_size
        self.node_color = node_color
        self.edge_color = edge_color
        self.linewidths = linewidths
        self.font_size = font_size
        self.output_dir = output_dir if output_dir else os.getcwd()
        self.output_filename = output_filename
        self.logger = get_logger(__name__)
        self.logger.info("Initialized StaticVisualizer.")

    def generate_graph(self) -> nx.Graph:
        """
        Converts the adjacency matrix into a NetworkX graph.

        Returns:
            nx.Graph: NetworkX graph constructed from the adjacency matrix.

        Raises:
            nx.NetworkXError: If the generated graph is empty.
        """
        self.logger.info("Generating NetworkX graph from adjacency matrix.")
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        self.logger.info(
            f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )

        if G.number_of_nodes() == 0:
            self.logger.error("Generated graph is empty.")
            raise nx.NetworkXError("Generated graph is empty.")

        return G

    def visualize(self, G: nx.Graph):
        """
        Generates and saves a static visualization of the network.

        Args:
            G (nx.Graph): NetworkX graph to visualize.
        """
        self.logger.info(f"Generating static visualization with layout: {self.layout}")

        if self.layout == "spring":
            pos = nx.spring_layout(G)
        elif self.layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif self.layout == "circular":
            pos = nx.circular_layout(G)
        elif self.layout == "random":
            pos = nx.random_layout(G)
        elif self.layout == "shell":
            pos = nx.shell_layout(G)
        else:
            self.logger.warning(
                f"Layout '{self.layout}' not recognized. Falling back to spring layout."
            )
            pos = nx.spring_layout(G)

        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(
            G, pos, node_size=self.node_size, node_color=self.node_color, alpha=0.7
        )
        nx.draw_networkx_edges(
            G, pos, edge_color=self.edge_color, width=self.linewidths, alpha=0.5
        )
        nx.draw_networkx_labels(G, pos, font_size=self.font_size, font_color="black")
        plt.title("Static Network Visualization", fontsize=15)
        plt.axis("off")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_filename)

        plt.tight_layout()
        plt.savefig(output_path, format="PNG")
        plt.close()
        self.logger.info(f"Static network visualization saved to {output_path}")
