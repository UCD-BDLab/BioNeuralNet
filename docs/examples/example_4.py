"""
Example 4: SmCCNet + PageRank Clustering + Visualization

This example demonstrates how to:
1. Construct a network using SmCCNet from multi-omics data.
2. Apply PageRank-based clustering to identify meaningful sub-networks.
3. Visualize the resulting network clusters using static or dynamic visualization tools.
"""

import os
import pandas as pd
import networkx as nx

from bioneuralnet.graph_generation import SmCCNet
from bioneuralnet.clustering import PageRank
from bioneuralnet.analysis import StaticVisualizer

def run_smccnet_pagerank_visualization_workflow():
    omics_proteins = pd.read_csv('input/proteins.csv', index_col=0)
    omics_metabolites = pd.read_csv('input/metabolites.csv', index_col=0)
    phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0).squeeze()
    omics_dfs = [omics_proteins, omics_metabolites]
    data_types = ['protein', 'metabolite']

    smccnet_instance = SmCCNet(
        phenotype_data=phenotype_data,
        omics_dfs=omics_dfs,
        data_types=data_types,
        kfold=5,
        summarization='PCA',
        seed=732
    )

    adjacency_matrix = smccnet_instance.run()
    adjacency_output_path = os.path.join(smccnet_instance.output_dir, 'adjacency_matrix.csv')
    adjacency_matrix.to_csv(adjacency_output_path)
    print(f"Adjacency matrix saved to {adjacency_output_path}")

    G = nx.from_pandas_adjacency(adjacency_matrix)

    pagerank_instance = PageRank(
        graph=G,
        omics_data=pd.concat(omics_dfs, axis=1),
        phenotype_data=phenotype_data,
        alpha=0.9,
        max_iter=100,
        tol=1e-6,
        k=0.9,
        output_dir='pagerank_output'
    )

    seed_nodes = ['node1', 'node2'] 

    try:
        results = pagerank_instance.run(seed_nodes=seed_nodes)
        print("PageRank Clustering Results:")
        print(results)
    except Exception as e:
        print(f"Error running PageRank clustering: {e}")
        return

    cluster_nodes = results.get('cluster_nodes', [])
    if cluster_nodes:
        subgraph = G.subgraph(cluster_nodes).copy()
        visualizer = StaticVisualizer(
            adjacency_matrix=nx.to_pandas_adjacency(subgraph),
            output_dir='visualization_output',
            output_filename='cluster_visualization.png'
        )
        G_sub = visualizer.generate_graph()
        visualizer.visualize(G_sub)
        print("Visualization saved to visualization_output/cluster_visualization.png")
    else:
        print("No cluster identified for visualization.")

if __name__ == "__main__":
    run_smccnet_pagerank_visualization_workflow()
