import pandas as pd
from bioneuralnet.external_tools import StaticVisualizer


def main():
    adjacency_matrix = pd.read_csv("input/adjacency_matrix.csv", index_col=0)

    static_vis = StaticVisualizer(
        adjacency_matrix=adjacency_matrix,
        layout="spring",
        node_size=300,
        node_color="skyblue",
        edge_color="gray",
        linewidths=1.0,
        font_size=10,
        output_dir="visualizations/static",
        output_filename="static_network.png",
    )

    G = static_vis.generate_graph()
    static_vis.visualize(G)
    print("Static visualization workflow completed successfully.")


if __name__ == "__main__":
    main()
