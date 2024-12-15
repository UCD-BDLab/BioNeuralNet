import os
from typing import Optional
import pandas as pd
import networkx as nx
from pyvis.network import Network
from ..utils.logger import get_logger


class DynamicVisualizer:
    """
    DynamicVisualizer Class for Generating Interactive Network Visualizations.
    
    Utilizes Pyvis to create and save interactive HTML visualizations of networks.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        layout: str = 'spring',
        notebook: bool = False,
        bgcolor: str = '#ffffff',
        font_color: str = 'black',
        output_dir: Optional[str] = None,
        output_filename: str = "dynamic_network.html",
        width: str = "100%",
        height: str = "800px"
    ):
        """
        Initializes the DynamicVisualizer instance.
        
        Args:
            adjacency_matrix (pd.DataFrame): Adjacency matrix representing the network.
            layout (str, optional): Layout algorithm for network visualization ('spring', 'hierarchical', etc.). Defaults to 'spring'.
            notebook (bool, optional): Whether to generate a notebook-compatible visualization. Defaults to False.
            bgcolor (str, optional): Background color of the visualization. Defaults to '#ffffff'.
            font_color (str, optional): Font color for node labels. Defaults to 'black'.
            output_dir (str, optional): Directory to save the visualization HTML file. If None, saves in the current directory. Defaults to None.
            output_filename (str, optional): Filename for the saved visualization HTML file. Defaults to "dynamic_network.html".
            width (str, optional): Width of the visualization. Defaults to "100%".
            height (str, optional): Height of the visualization. Defaults to "800px".
        """
        self.adjacency_matrix = adjacency_matrix
        self.layout = layout
        self.notebook = notebook
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.output_dir = output_dir if output_dir else os.getcwd()
        self.output_filename = output_filename
        self.width = width
        self.height = height
        self.logger = get_logger(__name__)
        self.logger.info("Initialized DynamicVisualizer.")

    def generate_graph(self) -> nx.Graph:
        """
        Converts the adjacency matrix into a NetworkX graph.
        
        Returns:
            nx.Graph: NetworkX graph constructed from the adjacency matrix.
        """
        self.logger.info("Generating NetworkX graph from adjacency matrix.")
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        self.logger.info(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def visualize(self, G: nx.Graph):
        """
        Generates and saves an interactive visualization of the network.
        
        Args:
            G (nx.Graph): NetworkX graph to visualize.
        """
        self.logger.info(f"Generating interactive visualization with layout: {self.layout}")
        
        net = Network(height=self.height, width=self.width, bgcolor=self.bgcolor, font_color=self.font_color, notebook=self.notebook)
        
        if self.layout == 'hierarchical':
            net.barnes_hut()
        elif self.layout == 'spring':
            net.force_atlas_2based()
        else:
            self.logger.warning(f"Layout '{self.layout}' not recognized. Using default layout.")
        
        net.from_nx(G)
        
        for node in net.nodes:
            node['title'] = node['id']
            node['label'] = node['id']
            node['color'] = 'skyblue'
        
        for edge in net.edges:
            edge['color'] = 'gray'
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_filename)
        
        net.show(output_path)
        self.logger.info(f"Interactive network visualization saved to {output_path}")
