"""
Test script for TCGA and STRING preprocessing pipeline with visualizations.

This script tests the preprocessing pipeline and generates visualizations
to help understand the data quality and structure.
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

from examples.preprocess_tcga_string import TCGAStringPreprocessor

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Output directory for visualizations
VIZ_DIR = Path(__file__).resolve().parents[1] / "visualization_results"
os.makedirs(VIZ_DIR, exist_ok=True)


def visualize_network(adjacency_matrix, node_mapping=None, title="PPI Network", max_nodes=500):
    """
    Visualize the network structure.
    """
    # Guarding against empty adjacency
    if adjacency_matrix is None or getattr(adjacency_matrix, 'shape', (0, 0))[0] == 0 or getattr(adjacency_matrix, 'shape', (0, 0))[1] == 0:
        print("Adjacency matrix is empty; skipping network visualization.")
        return

    print(f"Visualizing network with {adjacency_matrix.shape[0]} nodes...")

    G = nx.from_pandas_adjacency(adjacency_matrix)

    if G.number_of_nodes() == 0:
        print("Graph has no nodes; skipping network visualization.")
        return
    if G.number_of_edges() == 0:
        print("Graph has no edges; skipping network visualization.")
        return

    if G.number_of_nodes() > max_nodes:
        print(f"Network too large, sampling {max_nodes} nodes...")
        nodes = list(G.nodes())
        sampled_nodes = np.random.choice(nodes, max_nodes, replace=False)
        G = G.subgraph(sampled_nodes)

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

    # Drawing the network
    nx.draw_networkx_nodes(G, pos,
                          node_size=[50 + 10 * degrees[node] for node in G.nodes()],
                          node_color='skyblue',
                          alpha=0.8)

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='gray')

    # Only labelling nodes with high degree for readability
    high_degree_nodes = {node: label for node, label in node_labels.items()
                        if degrees[node] > np.percentile(list(degrees.values()), 90)}

    nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, font_size=8)

    plt.title(f"{title} (showing {G.number_of_nodes()} nodes)")
    plt.axis('off')

    plt.tight_layout()
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Network visualization saved to {VIZ_DIR / filename}")


def visualize_feature_distribution(features_df, title="Feature Distribution"):
    """
    Visualize the distribution of features.
    """
    print("Visualizing feature distributions...")

    numeric_df = features_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        print("No numeric features available; skipping feature distribution visualization.")
        return
    numeric_df = numeric_df.dropna(axis=1, how='all')
    if numeric_df.empty:
        print("All numeric features are NaN; skipping feature distribution visualization.")
        return

    if numeric_df.shape[1] > 20:
        sampled_features = np.random.choice(numeric_df.columns, 20, replace=False)
        features_to_plot = numeric_df[sampled_features]
    else:
        features_to_plot = numeric_df

    if features_to_plot.empty:
        print("No features to plot after sampling; skipping feature distribution visualization.")
        return

    plt.figure(figsize=(14, 10))
    sns.boxplot(data=features_to_plot)
    plt.xticks(rotation=90)
    plt.title(f"{title} (sample of {features_to_plot.shape[1]} features)")
    plt.ylabel("Value")
    plt.xlabel("Feature")

    plt.tight_layout()
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Feature distribution visualization saved to {VIZ_DIR / filename}")

    # Also creating a heatmap of feature correlations
    corr = features_to_plot.corr()
    if corr.empty:
        print("Correlation matrix is empty; skipping correlation heatmap.")
    else:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title(f"Feature Correlation Heatmap")

        plt.tight_layout()
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

    data = features_df.select_dtypes(include=[np.number]).copy()
    if data.empty:
        print("No numeric features available for embedding; skipping visualization.")
        return
    data = data.fillna(0)

    n_samples, n_features = data.shape

    if method.lower() == 'tsne':
        if n_samples < 3 or n_features < 1:
            print("Insufficient data for t-SNE; falling back to PCA.")
            n_components_pca = max(1, min(2, n_features, n_samples))
            if n_components_pca < 1:
                print("Not enough data for PCA; skipping embedding.")
                return
            reducer = PCA(n_components=n_components_pca, random_state=42)
            embedding = reducer.fit_transform(data)
            method_name = "PCA"
        else:
            safe_perp = max(2, min(30, n_samples - 1))
            reducer = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto', perplexity=safe_perp)
            embedding = reducer.fit_transform(data)
            method_name = "t-SNE"
    else:
        n_components_pca = max(1, min(2, n_features, n_samples))
        if n_components_pca < 1:
            print("Not enough data for PCA; skipping embedding.")
            return
        reducer = PCA(n_components=n_components_pca, random_state=42)
        embedding = reducer.fit_transform(data)
        method_name = "PCA"

    # Creating DataFrame for plotting
    if embedding.shape[1] == 1:
        embedding = np.hstack([embedding, np.zeros((embedding.shape[0], 1))])
    embedding_df = pd.DataFrame(embedding, columns=['Component 1', 'Component 2'])
    embedding_df['Sample'] = data.index

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=embedding_df, x='Component 1', y='Component 2', s=100, alpha=0.7)

    label_step = max(1, len(embedding_df) // 20)
    for i, sample in enumerate(embedding_df['Sample']):
        if i % label_step == 0:
            plt.annotate(sample, (embedding_df.iloc[i, 0], embedding_df.iloc[i, 1]),
                        fontsize=8, alpha=0.7)

    plt.title(f"{title} ({method_name})")

    plt.tight_layout()
    filename = f"{title.lower().replace(' ', '_')}_{method.lower()}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Embedding visualization saved to {VIZ_DIR / filename}")


def visualize_node_feature_importance(node_features, node_mapping=None, title="Node Feature Importance"):
    """
    Visualize the importance of features for each node.

    Args:
        node_features: Dictionary mapping sample IDs to node features
        node_mapping: Mapping from node IDs to gene symbols
        title: Title of the plot
    """
    print("Visualizing node feature importance...")

    if not isinstance(node_features, dict) or len(node_features) == 0:
        print("No node features to visualize")
        return

    # Calculating average feature values across samples
    sample_keys = list(node_features.keys())
    if not sample_keys:
        print("No node features to visualize")
        return

    first = node_features[sample_keys[0]]
    if isinstance(first, pd.DataFrame):
        first = first.values
    elif isinstance(first, pd.Series):
        first = first.values.reshape(-1, 1)

    if first is None or np.size(first) == 0:
        print("Node features are empty; skipping visualization.")
        return

    num_nodes = first.shape[0]
    num_features = first.shape[1] if first.ndim > 1 else 1
    if num_nodes < 1 or num_features < 1:
        print("Insufficient node feature dimensions; skipping visualization.")
        return

    avg_features = np.zeros((num_nodes, num_features), dtype=float)
    count = 0
    for sample_id, features in node_features.items():
        arr = features
        if isinstance(arr, pd.DataFrame):
            arr = arr.values
        elif isinstance(arr, pd.Series):
            arr = arr.values.reshape(-1, 1)
        if arr is None or arr.shape != (num_nodes, num_features):
            continue
        avg_features += np.nan_to_num(arr)
        count += 1
    if count == 0:
        print("No valid node feature matrices to aggregate; skipping visualization.")
        return
    avg_features /= max(1, count)

    # Calculating feature importance as variance across nodes
    feature_importance = np.var(avg_features, axis=0)

    # Creating the plot for feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel("Feature Index")
    plt.ylabel("Variance (Importance)")
    plt.title(f"{title} (variance across nodes)")

    plt.tight_layout()
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(VIZ_DIR / filename, dpi=300)
    plt.close()

    print(f"Node feature importance visualization saved to {VIZ_DIR / filename}")

    # Also creating a heatmap of top nodes by feature importance
    node_variance = np.var(avg_features, axis=1) if avg_features.size > 0 else np.array([])
    if node_variance.size == 0 or num_nodes == 0:
        print("No variability across nodes; skipping top nodes heatmap.")
        return
    top_count = min(20, num_nodes)
    top_nodes_idx = np.argsort(node_variance)[-top_count:]
    if top_nodes_idx.size == 0:
        print("No top nodes to display; skipping heatmap.")
        return
    top_nodes_features = avg_features[top_nodes_idx]
    if top_nodes_features.size == 0:
        print("Top nodes feature matrix is empty; skipping heatmap.")
        return

    # Creating labels for the heatmap
    if node_mapping is not None and hasattr(node_mapping, 'index'):
        node_ids = list(node_mapping.index)
        row_labels = [node_mapping.loc[node_ids[idx], 'gene_symbol'] if idx < len(node_ids) else f"Node {idx}"
                      for idx in top_nodes_idx]
    else:
        row_labels = [f"Node {idx}" for idx in top_nodes_idx]

    col_labels = [f"Feature {i}" for i in range(num_features)]

    # Creating the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_nodes_features, cmap='viridis', yticklabels=row_labels, xticklabels=col_labels)
    plt.title("Top Nodes by Feature Variance")

    # Saving the heatmap
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "top_nodes_feature_heatmap.png", dpi=300)
    plt.close()

    print(f"Top nodes feature heatmap saved to {VIZ_DIR / 'top_nodes_feature_heatmap.png'}")


def run_test(use_real_data=False, cancer_type="BRCA"):
    """
    Run the preprocessing pipeline and generate visualizations.

    Args:
        use_real_data: Whether to attempt to download real data
        cancer_type: TCGA cancer type to use
    """
    print(f"Testing preprocessing pipeline with {'real' if use_real_data else 'sample'} data...")

    output_dir = Path(__file__).resolve().parents[1] / "test_processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Initializing preprocessor
    preprocessor = TCGAStringPreprocessor(
        tcga_dir=None,
        string_file=None,
        output_dir=output_dir,
        string_cache_dir=Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets" / cancer_type.lower() / "string",
        cancer_type=cancer_type,
        omics_types=["gene_expression"],
        download_if_missing=True
    )

    # Running the pipeline
    print("Running preprocessing pipeline...")
    features_df, adjacency_matrix = preprocessor.run_pipeline()

    print("\nPreprocessing complete. Generating visualizations...")

    node_mapping_file = output_dir / "node_mapping.csv"
    if os.path.exists(node_mapping_file):
        node_mapping = pd.read_csv(node_mapping_file)
        node_mapping = node_mapping.set_index('string_id')
    else:
        if hasattr(preprocessor, 'gene_id_map') and preprocessor.gene_id_map:
            node_mapping = pd.DataFrame({
                'string_id': list(preprocessor.gene_id_map.keys()),
                'gene_symbol': list(preprocessor.gene_id_map.values())
            }).set_index('string_id')
        else:
            node_mapping = None

    # Generating visualizations
    visualize_network(adjacency_matrix, node_mapping, title=f"STRING PPI Network ({cancer_type})")
    visualize_feature_distribution(features_df, title=f"Node Features ({cancer_type})")
    visualize_embedding(features_df, method='tsne', title=f"Sample Embedding ({cancer_type})")
    visualize_embedding(features_df, method='pca', title=f"Sample Embedding ({cancer_type})")

    # Visualizing node features if available
    if hasattr(preprocessor, 'normalized_features') and preprocessor.normalized_features:
        visualize_node_feature_importance(preprocessor.normalized_features, node_mapping,
                                         title=f"Node Feature Importance ({cancer_type})")

    print("\nAll visualizations generated successfully!")
    print(f"Results saved to {VIZ_DIR}")

    return features_df, adjacency_matrix


if __name__ == "__main__":
    # Running the test with sample data (faster)
    features_df, adjacency_matrix = run_test(use_real_data=False)

    # Uncomment to run with real data (requires internet connection and takes longer)
    # features_df, adjacency_matrix = run_test(use_real_data=True, cancer_type="BRCA")
