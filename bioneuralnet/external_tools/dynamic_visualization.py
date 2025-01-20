import os
from typing import Optional
import pandas as pd
import networkx as nx
import pyvis
from pyvis.network import Network
from jinja2 import Environment, FileSystemLoader
from ..utils.logger import get_logger


class DynamicVisualizer:
    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        layout: str = "spring",
        notebook: bool = False,
        bgcolor: str = "#ffffff",
        font_color: str = "black",
        output_dir: Optional[str] = None,
        output_filename: str = "dynamic_network.html",
        width: str = "100%",
        height: str = "800px",
    ):
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
        self.logger.info("Generating NetworkX graph from adjacency_matrix.")
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        self.logger.info(
            f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )
        if G.number_of_nodes() == 0:
            self.logger.error("Generated graph is empty.")
            raise nx.NetworkXError("Generated graph is empty.")
        return G

    def visualize(self, G: nx.Graph):
        self.logger.info(
            f"Generating interactive visualization with layout: {self.layout}"
        )

        templates_dir = os.path.join(os.path.dirname(pyvis.__file__), "templates")
        env = Environment(loader=FileSystemLoader(templates_dir))
        template = env.get_template("template.html")

        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            notebook=False,
        )
        net.template = template

        if self.layout == "hierarchical":
            net.barnes_hut()
        elif self.layout == "spring":
            net.force_atlas_2based()
        else:
            self.logger.warning(
                f"Layout '{self.layout}' not recognized. Falling back to spring layout."
            )

        net.from_nx(G)

        for node in net.nodes:
            node["title"] = node["id"]
            node["label"] = node["id"]
            node["color"] = "skyblue"

        for edge in net.edges:
            edge["color"] = "gray"

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_filename)
        net.show(output_path)
        self.logger.info(f"Interactive network visualization saved to {output_path}")
