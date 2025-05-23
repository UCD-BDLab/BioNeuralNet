import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, cast
from bioneuralnet.utils import get_logger
from .correlation import cluster_correlation
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# try:
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     from sklearn.manifold import TSNE

# except ImportError:
#     raise ImportError("Please install the required packages for plotting: pip install matplotlib")

logger = get_logger(__name__)

def plot_variance_distribution(df: pd.DataFrame, bins: int = 50):
    """
    Compute the variance for each feature (column) in the DataFrame and plot
    a histogram of these variances.

    Parameters:

        df (pd.DataFrame): Input data.
        bins (int): Number of bins for the histogram.

    Returns:

        matplotlib.figure.Figure: Generated figure.
    """
    variances = df.var()
    logger.info("Computed variances for each feature.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(variances, bins=bins, edgecolor='black')
    ax.set_title("Distribution of Feature Variances")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Frequency")

    logger.info("Variance distribution plot generated.")
    return fig

def plot_variance_by_feature(df: pd.DataFrame):
    """
    Plot the variance for each feature against its index or name.

    Parameters:

        df (pd.DataFrame): Input data.

    Returns:

        matplotlib.figure.Figure: Generated figure.
    """
    variances = df.var()
    logger.info("Computed variances for each feature for index plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(variances.index, variances.values, 'o', markersize=4)
    ax.set_title("Variance per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Variance")
    ax.tick_params(axis='x', rotation=90)

    logger.info("Variance vs. feature index plot generated.")
    return fig

def plot_performance_three(raw_score, gnn_score, other_score, labels=["Raw","GNN","Other"], title="Performance Comparison", filename=None):
    """
    Bar plot comparing performance for raw omics, GNN-enriched omics, and one other method.
    """
    if len(raw_score) != 2 or len(gnn_score) != 2 or len(other_score) != 2:
        raise ValueError("Scores must be tuples of (mean, std)")
    scores = [raw_score[0], gnn_score[0], other_score[0]]
    errors = [raw_score[1], gnn_score[1], other_score[1]]

    x = np.arange(len(scores))
    width = 0.23

    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(x, scores, width, yerr=errors, capsize=3,
                  color=["#4E79A7", "#F28E2B", "#76B7B2"], alpha=0.95, linewidth=0)

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        err = errors[i]
        ax.text(bar.get_x() + bar.get_width() / 2, height + err + 0.015,
                f"{height:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.85)

    if filename:
        plt.savefig(str(filename), dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename}")

    plt.show()

def plot_performance(embedding_result, raw_rf_acc, title="Performance Comparison", filename=None, metric_name="Accuracy", labels=None):
    """
    A minimal bar plot comparing two performance metrics. e.g., raw vs embeddings based performance.

    Parameters:

        embedding_result: tuple or dict containing (mean, std) of the metric for the model using embeddings.
        raw_rf_acc: tuple or dict containing (mean, std) of the metric for the baseline model.
        title: string, title of the plot.
        filename: string or Path, if provided, the plot will be saved to this file.
        metric_name: string, name of the metric to display on the Y-axis (e.g., "Accuracy").
        labels: iterable of two strings (default: ["Label 1", "Label 2"]).

    Returns:

        None. Displays (and optionally saves) the bar plot.
    """
    def parse_score(x):
        if isinstance(x, dict):
            return x.get("accuracy", 0), x.get("std", 0)
        elif isinstance(x, tuple) and len(x) == 2:
            return x
        else:
            return float(x), 0

    embed_acc, embed_std = parse_score(embedding_result)
    raw_acc, raw_std = parse_score(raw_rf_acc)

    if labels is None:
        labels = ["Label 1", "Label 2"]
    if len(labels) != 2:
        raise ValueError("labels must be a sequence of length 2.")

    scores = [raw_acc, embed_acc]
    errors = [raw_std, embed_std]
    x = np.arange(len(scores))
    width = 0.23

    fig, ax = plt.subplots(figsize=(3.2, 4))
    bars = ax.bar(x, scores, width,yerr=errors, capsize=2,color=["#4E79A7", "#F28E2B"],alpha=0.95, linewidth=0)

    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        h   = bar.get_height()
        err = errors[i]
        ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.015, f"{h:.3f}",ha="center", va="bottom",fontsize=10, fontweight="bold")

    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.88)

    if filename:
        plt.savefig(str(filename), dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename}")

    plt.show()


def plot_embeddings(embeddings, node_labels=None):
    """
    Plot the embeddings in 2D space using t-SNE.

    Parameters:

        embeddings (array-like): High-dimensional embedding data.
        node_labels (array-like or DataFrame, optional): Labels for the nodes to color the points.

    """
    X = np.array(embeddings)

    perplexity = min(30, X.shape[0] - 1)
    if perplexity < 1:
        logger.info(f"Skipping plot: not enough samples ({X.shape[0]}) for TSNE.")
        return
    reducer = TSNE(n_components=2, init="pca", perplexity=perplexity)

    X_reduced = reducer.fit_transform(X)

    if node_labels is None:
        c_values = np.zeros(X.shape[0])
    elif hasattr(node_labels, "iloc"):
        node_labels= node_labels.to_frame(name="phenotype")
        c_values = np.array(node_labels.iloc[:, 0], dtype=float).flatten()
    else:
        c_values = np.array(node_labels, dtype=float).flatten()

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        X_reduced[:, 0], X_reduced[:, 1],
        c=c_values,
        cmap="viridis",
        s=60,
        alpha=0.9,
        edgecolor="k"
    )

    ax.invert_yaxis()
    ax.set_title(f"Embeddings in 2D space from {embeddings.shape[1]}D")

    fig.tight_layout()
    plt.show()


def plot_network(adjacency_matrix, weight_threshold=0.0, show_labels=False, show_edge_weights=False):
    """
    Plots a network graph from an adjacency matrix with improved visualization.
    Also adds a summary table mapping node indexes to actual gene names.

    Parameters:

        adjacency_matrix (pd.DataFrame): The adjacency matrix of the network.
        weight_threshold (float): Minimum weight to keep an edge (default: 0.0).
        show_labels (bool): Whether to show node labels.
        show_edge_weights (bool): Whether to show edge weights

    Returns:

        pd.DataFrame: Mapping of node indexes to actual gene names.
    """
    full_G = nx.from_pandas_adjacency(adjacency_matrix)
    total_nodes = full_G.number_of_nodes()
    total_edges = full_G.number_of_edges()

    G = full_G.copy()
    for _, _, d in G.edges(data=True):
        d["abs_weight"] = abs(d.get("weight", 0.0))

    if weight_threshold > 0:
        edges_to_remove = []

        for u, v, d in G.edges(data=True):
            raw_w = d.get('weight', 0)
            abs_w = abs(raw_w)
            d['abs_weight'] = abs_w
            if abs_w < weight_threshold:
                edges_to_remove.append((u, v))

        G.remove_edges_from(edges_to_remove)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    current_nodes = list(G.nodes())
    current_edges = G.number_of_edges()

    index_mapping = {}
    for i, node in enumerate(current_nodes):
        index_mapping[node] = i + 1

    indexed_labels = {}
    for node in current_nodes:
        indexed_labels[node] = str(index_mapping[node])

    degrees = {}
    for node, degree in G.degree():
        degrees[node] = degree

    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = []
    for node in G.nodes():
        node_sizes.append(150 + (degrees[node] / max_degree) * 300)

    edge_weights = []
    for u, v, d in G.edges(data=True):
        edge_weights.append(d['abs_weight'])

    edge_widths = []
    if edge_weights:
        min_w = edge_weights[0]
        max_w = edge_weights[0]
        for w in edge_weights:
            if w < min_w:
                min_w = w
            if w > max_w:
                max_w = w

        span = max_w - min_w if max_w > min_w else 1e-6

        for w in edge_weights:
            width = 2 + 4 * (w - min_w) / span
            edge_widths.append(width)
    else:
        edge_widths = []

    pos = nx.kamada_kawai_layout(G, weight="abs_weight")
    fig, ax_graph = plt.subplots(figsize=(14, 8))

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="gold", edgecolors="black", linewidths=1.5, alpha=0.9, ax=ax_graph)
    nx.draw_networkx_edges(G, pos, alpha=0.9, width=edge_widths, edge_color="black", ax=ax_graph)

    if show_edge_weights and edge_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')

        formatted_edge_labels = {}
        for edge, weight in edge_labels.items():
            formatted_edge_labels[edge] = f"{weight:.4f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_size=9, ax=ax_graph)

    if show_labels:
        nx.draw_networkx_labels(G, pos, labels=indexed_labels, font_size=11, font_color="black", ax=ax_graph)

    ax_graph.set_xticks([])
    ax_graph.set_yticks([])
    ax_graph.set_frame_on(False)

    ax_graph.set_title("Network Visualization", fontsize=16)
    ax_graph.axis("off")

    summary_text = f"""
    Full Cluster Nodes: {total_nodes}
    Full Cluster Edges: {total_edges}
    Filtered Nodes: {len(current_nodes)}
    Filtered Edges: {current_edges}
    """

    ax_graph.text(0.9, 1.05, summary_text, transform=ax_graph.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()

    mapping_data = []
    for node in current_nodes:
        mapping_data.append((index_mapping[node], node, degrees[node]))

    mapping_df = pd.DataFrame(mapping_data, columns=["Index", "Omic", "Degree"])
    mapping_df = mapping_df.sort_values(by="Degree", ascending=False).set_index("Index")

    return mapping_df

def compare_clusters(louvain_clusters: list, smccnet_clusters: list, pheno: pd.DataFrame, omics_merged: pd.DataFrame, label1: str = "Hybrid Louvain", label2: str = "SmCCNet"):
    """
    Compare clusters from two methods by computing the correlation for each induced subnetwork.
    Both inputs are expected to be lists of pandas DataFrames. If the lists have different lengths,
    only the first min(n, m) clusters are compared.

    Parameters:

        louvain_clusters (list of pd.DataFrame): Each DataFrame represents an induced subnetwork (e.g. from Hybrid Louvain).
        smccnet_clusters (list of pd.DataFrame): Each DataFrame represents an induced subnetwork (e.g. from SMCCNET).
        pheno (pd.DataFrame): Phenotype data (the first column is used).
        omics_merged (pd.DataFrame): Full omics data (used to re-index SMCCNet clusters).
        label1 (str): Label for the first method.
        label2 (str): Label for the second method.

    Returns:

        pd.DataFrame: Results table with cluster indices, sizes, and phenotype correlations.

    Plots:

        A line plot comparing the correlation of PC1 vs phenotype for each method.
        Louvain and SMCCNet curves are shown with their cluster sizes annotated.
    """

    smccnet_clusters_fixed = []

    i = 0
    while i < len(smccnet_clusters):
        cluster_df = smccnet_clusters[i]
        valid_genes = []

        j = 0
        while j < len(cluster_df.index):
            gene = cluster_df.index[j]
            if gene in omics_merged.columns:
                valid_genes.append(gene)
            j += 1

        if len(valid_genes) > 0:
            sample_level_data = omics_merged[valid_genes]
            smccnet_clusters_fixed.append(sample_level_data)
        i += 1

    louvain_sizes = []
    i = 0
    while i < len(louvain_clusters):
        df = louvain_clusters[i]
        size = df.shape[1]
        louvain_sizes.append((i, size))
        i += 1

    smccnet_sizes = []
    i = 0
    while i < len(smccnet_clusters_fixed):
        df = smccnet_clusters_fixed[i]
        size = df.shape[1]
        smccnet_sizes.append((i, size))
        i += 1

    def get_size(pair):
        return pair[1]

    louvain_sizes.sort(key=get_size, reverse=True)
    smccnet_sizes.sort(key=get_size, reverse=True)

    if len(louvain_sizes) < len(smccnet_sizes):
        k = len(louvain_sizes)
    else:
        k = len(smccnet_sizes)

    top_louvain_indices = []
    i = 0
    while i < k:
        top_louvain_indices.append(louvain_sizes[i][0])
        i += 1

    top_smccnet_indices = []
    i = 0
    while i < k:
        top_smccnet_indices.append(smccnet_sizes[i][0])
        i += 1

    top_louvain_clusters = []
    i = 0
    while i < len(top_louvain_indices):
        idx = top_louvain_indices[i]
        top_louvain_clusters.append(louvain_clusters[idx])
        i += 1

    top_smccnet_clusters = []
    i = 0
    while i < len(top_smccnet_indices):
        idx = top_smccnet_indices[i]
        top_smccnet_clusters.append(smccnet_clusters_fixed[idx])
        i += 1

    louvain_clusters = top_louvain_clusters
    smccnet_clusters_fixed = top_smccnet_clusters

    results = []

    for i, (df_louvain, df_smccnet) in enumerate(zip(louvain_clusters, smccnet_clusters_fixed), start=1):
        size_louvain, corr_louvain = cluster_correlation(df_louvain, pheno)
        size_smccnet, corr_smccnet = cluster_correlation(df_smccnet, pheno)

        if corr_louvain is not None and corr_smccnet is not None:
            results.append((f"Cluster_{i}", size_louvain, corr_louvain, size_smccnet, corr_smccnet))

    df_results = pd.DataFrame(results, columns=["Cluster", "Louvain Size", "Louvain Correlation",
                                                "SMCCNET Size", "SMCCNET Correlation"])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_results.index + 1, df_results["Louvain Correlation"], marker="o", linestyle="-",
            label=label1, color="blue")
    ax.plot(df_results.index + 1, df_results["SMCCNET Correlation"], marker="s", linestyle="--",
            label=label2, color="red")

    for i, row in df_results.iterrows():
        ax.text(i + 1, row["Louvain Correlation"] + 0.05,
                f"{row['Louvain Size']}", ha="center", fontsize=10,
                color="blue", fontweight="bold", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.text(i + 1, row["SMCCNET Correlation"] + 0.05,
                f"{row['SMCCNET Size']}", ha="center", fontsize=10,
                color="red", fontweight="bold", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xticks(range(1, len(df_results) + 1))
    ax.set_xlabel("Cluster Index")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Cluster correlation:{label1} vs {label2}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout(pad=3)
    plt.show()

    return df_results

def plot_multiple_metrics(
    metrics: dict[str, dict[str, tuple[float, float]]],
    title_map: Optional[dict[str, str]] = None,
    ylabel_map: Optional[dict[str, str]] = None,
    filename: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot multiple grouped performance metrics as side-by-side bar plots.

    Parameters:

        metrics (dict): A nested dictionary of the form:{metric_name: {group_name: (mean, std)}}.
        title_map (dict, optional): Maps each metric name to a custom plot title.
        ylabel_map (dict, optional): Maps each metric name to a Y-axis label.
        filename (str or Path, optional): If provided, saves the figure to this path.

    Returns:

        None. Displays or shows a matplotlib figure with grouped metric comparisons.

    """
    logger.info(f"Plotting multiple metrics: {list(metrics.keys())}")
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (metric, sc) in zip(axes, metrics.items()):
        groups = list(sc.keys())
        sublabels = [""]

        means_list: list[list[float]] = []
        errs_list: list[list[float]] = []

        for g in groups:
            row_m: list[float] = []
            row_e: list[float] = []
            for s in sublabels:
                m, e = sc[g]
                row_m.append(m)
                row_e.append(e)
            means_list.append(row_m)
            errs_list.append(row_e)

        means = cast(np.ndarray, np.array(means_list, dtype=float))
        errs  = cast(np.ndarray, np.array(errs_list,  dtype=float))
        ind = np.arange(len(groups))
        total = 0.7
        width = total / len(sublabels)

        for i in range(len(sublabels)):
            x = ind + i * width
            y = means[:, i]
            yerr = errs[:, i]

            bars = ax.bar(x, y, width, yerr=yerr, capsize=3,color=["#4E79A7", "#F28E2B"], label=sublabels[i])
            for bar in bars:
                height = bar.get_height()
                bar_x = bar.get_x() + bar.get_width() / 2
                bar_y = height + 0.01
                ax.text(bar_x, bar_y, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(ind + total / 2 - width / 2)
        ax.set_xticklabels(groups, fontsize=11)

        title = title_map.get(metric, metric) if title_map else metric
        ylabel = ylabel_map.get(metric, metric) if ylabel_map else metric

        ax.set_title(title, fontsize=14, pad=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if "Accuracy" in ylabel or "F1" in ylabel:
            ax.set_ylim(0, 1.3)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout(pad=2.0)

    if filename:
        fig.savefig(str(filename), dpi=300, bbox_inches="tight")
        logger.info(f"Saved combined figure to {filename}")
    else:
        plt.show()

    plt.close(fig)
