"""
Test script with synthetic data for visualization.

This script generates synthetic data and creates visualizations
to demonstrate the preprocessing pipeline functionality.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

VIZ_DIR = Path(__file__).resolve().parents[1] / "visualization_results"
os.makedirs(VIZ_DIR, exist_ok=True)


def generate_synthetic_data(num_nodes=100, num_samples=20, num_features=50):
    """
    Generate synthetic data for testing.

    Args:
        num_nodes: Number of nodes in the network
        num_samples: Number of samples
        num_features: Number of features per node

    Returns:
        features_df: DataFrame with features
        adjacency_matrix: Adjacency matrix of the network
        node_mapping: Mapping from node IDs to gene symbols
    """
    print(f"Generating synthetic data with {num_nodes} nodes, {num_samples} samples, and {num_features} features...")

    features = np.random.normal(0, 1, (num_samples, num_nodes))
    sample_ids = [f"SAMPLE_{i}" for i in range(num_samples)]
    node_ids = [f"NODE_{i}" for i in range(num_nodes)]
    features_df = pd.DataFrame(features, index=sample_ids, columns=node_ids)

    G = nx.barabasi_albert_graph(num_nodes, 3, seed=42)

    node_map = {i: node_ids[i] for i in range(num_nodes)}
    G = nx.relabel_nodes(G, node_map)
    adjacency_matrix = nx.to_pandas_adjacency(G)

    gene_symbols = [f"GENE_{chr(65 + i % 26)}{i//26}" for i in range(num_nodes)]
    node_mapping = pd.DataFrame({
        'node_id': node_ids,
        'gene_symbol': gene_symbols
    })
    node_mapping = node_mapping.set_index('node_id')

    return features_df, adjacency_matrix, node_mapping


def visualize_network(adjacency_matrix, node_mapping=None, title="Synthetic Network"):
    """
    Visualize the network structure.

    Args:
        adjacency_matrix: Adjacency matrix of the network
        node_mapping: Mapping from node IDs to gene symbols
        title: Title of the plot
    """
    print(f"Visualizing network with {adjacency_matrix.shape[0]} nodes...")

    # Creating a graph from the adjacency matrix
    G = nx.from_pandas_adjacency(adjacency_matrix)

    # If the graph is too large, taking a subgraph of the most connected nodes
    if G.number_of_nodes() > 50:
        print(f"Network too large, showing top 50 nodes by degree...")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:50]
        G = G.subgraph(top_nodes)

    if node_mapping is not None:
        node_labels = {}
        for node in G.nodes():
            if node in node_mapping.index:
                node_labels[node] = node_mapping.loc[node, 'gene_symbol']
            else:
                node_labels[node] = node
    else:
        node_labels = {node: node for node in G.nodes()}

    degrees = dict(G.degree())

    plt.figure(figsize=(14, 14))

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos,
                          node_size=[50 + 10 * degrees[node] for node in G.nodes()],
                          node_color='skyblue',
                          alpha=0.8)

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='gray')

    high_degree_nodes = {node: label for node, label in node_labels.items()
                        if degrees[node] > np.percentile(list(degrees.values()), 75)}

    nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, font_size=8)

    plt.title(f"{title} (showing {G.number_of_nodes()} nodes)")
    plt.axis('off')

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Network visualization saved to {VIZ_DIR / filename}")


def visualize_feature_distribution(features_df, title="Feature Distribution"):
    """
    Visualize the distribution of features.

    Args:
        features_df: DataFrame containing features
        title: Title of the plot
    """
    print("Visualizing feature distributions...")

    if features_df.shape[1] > 20:
        sampled_features = np.random.choice(features_df.columns, 20, replace=False)
        features_to_plot = features_df[sampled_features]
    else:
        features_to_plot = features_df

    plt.figure(figsize=(14, 10))

    sns.boxplot(data=features_to_plot)
    plt.xticks(rotation=90)
    plt.title(f"{title} (sample of {features_to_plot.shape[1]} features)")
    plt.ylabel("Value")
    plt.xlabel("Feature")

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Feature distribution visualization saved to {VIZ_DIR / filename}")

    plt.figure(figsize=(12, 10))
    corr = features_to_plot.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f')
    plt.title(f"Feature Correlation Heatmap")

    plt.savefig(VIZ_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

    print(f"Feature correlation heatmap saved to {VIZ_DIR / 'feature_correlation_heatmap.png'}")


def visualize_embedding(features_df, method='tsne', title="Feature Embedding"):
    """
    Visualize the embedding of features using dimensionality reduction.

    Args:
        features_df: DataFrame containing features
        method: Dimensionality reduction method ('tsne' or 'pca')
        title: Title of the plot
    """
    print(f"Visualizing feature embedding using {method.upper()}...")

    data = features_df.T

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        method_name = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        method_name = "PCA"

    embedding_df = pd.DataFrame(embedding, columns=['Component 1', 'Component 2'])
    embedding_df['Sample'] = data.index

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=embedding_df, x='Component 1', y='Component 2', s=100, alpha=0.7)

    for i, sample in enumerate(embedding_df['Sample']):
        if i % max(1, len(embedding_df) // 20) == 0:  # Label every ~5% of points
            plt.annotate(sample, (embedding_df.iloc[i, 0], embedding_df.iloc[i, 1]),
                        fontsize=8, alpha=0.7)

    plt.title(f"{title} ({method_name})")

    filename = f"{title.lower().replace(' ', '_')}_{method.lower()}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Embedding visualization saved to {VIZ_DIR / filename}")


def run_test():
    """
    Run the test with synthetic data and generate visualizations.
    """
    print("Testing with synthetic data...")

    features_df, adjacency_matrix, node_mapping = generate_synthetic_data(
        num_nodes=100,
        num_samples=20,
        num_features=50
    )

    print("\nData generation complete. Generating visualizations...")

    visualize_network(adjacency_matrix, node_mapping, title="Synthetic PPI Network")
    visualize_feature_distribution(features_df, title="Synthetic Node Features")
    visualize_embedding(features_df, method='tsne', title="Synthetic Sample Embedding")
    visualize_embedding(features_df, method='pca', title="Synthetic Sample Embedding")

    print("\nAll visualizations generated successfully!")
    print(f"Results saved to {VIZ_DIR}")

    return features_df, adjacency_matrix


if __name__ == "__main__":
    features_df, adjacency_matrix = run_test()
