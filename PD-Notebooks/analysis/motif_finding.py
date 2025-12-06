"""
Motif finding and frequent subgraph mining for PD gene-gene networks.

This module provides functions for:
- Extracting k-node subgraphs (motifs) from graphs
- Finding frequent motifs
- Comparing motif frequencies between PD and Control graphs
- Statistical significance testing with null models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
from scipy import stats

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MotifResult:
    """
    Container for motif finding results.

    Attributes
    ----------
    motif_id : str
        Unique identifier for the motif (canonical form).
    nodes : List[str]
        List of gene names in the motif.
    frequency : int
        Number of times motif appears in the graph.
    z_score : float
        Z-score compared to null model.
    p_value : float
        P-value from significance test.
    """
    motif_id: str
    nodes: List[str]
    frequency: int
    z_score: float
    p_value: float


def extract_k_node_subgraphs(
    G: nx.Graph,
    k: int = 3,
    max_subgraphs: Optional[int] = None,
    sample_nodes: Optional[List] = None,
) -> List[nx.Graph]:
    """
    Extract all k-node connected subgraphs from a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : int, default=3
        Number of nodes in subgraphs (motif size).
    max_subgraphs : int, optional
        Maximum number of subgraphs to extract (for large graphs).
        If None, extracts all.
    sample_nodes : List, optional
        If provided, only extract subgraphs starting from these nodes.
        If None, uses all nodes.

    Returns
    -------
    List[nx.Graph]
        List of k-node subgraphs.
    """
    if sample_nodes is None:
        sample_nodes = list(G.nodes())

    subgraphs: List[nx.Graph] = []
    seen = set()

    for start_node in sample_nodes:
        if start_node not in G:
            continue

        # BFS to find all k-node subgraphs starting from this node
        queue = [(start_node, [start_node])]

        while queue and (max_subgraphs is None or len(subgraphs) < max_subgraphs):
            current_node, path = queue.pop(0)

            if len(path) == k:
                # Create subgraph from path
                nodes = tuple(sorted(path))
                if nodes not in seen:
                    subgraph = G.subgraph(path).copy()
                    if nx.is_connected(subgraph):
                        subgraphs.append(subgraph)
                        seen.add(nodes)
                continue

            # Add neighbors to path
            for neighbor in G.neighbors(current_node):
                if neighbor not in path:
                    new_path = path + [neighbor]
                    if len(new_path) <= k:
                        queue.append((neighbor, new_path))

    return subgraphs


def canonical_motif_form(G: nx.Graph) -> str:
    """
    Convert a graph to a canonical string representation for isomorphism checking.

    Uses graph structure (degree sequence and edge count) to create a unique identifier
    that is invariant to node labels.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    str
        Canonical string representation of the motif.
    """
    # Get sorted degree sequence (invariant to node labels)
    degrees = tuple(sorted([G.degree(n) for n in G.nodes()]))

    # For small graphs, we can use degree sequence + edge count
    # This works well for k=2, k=3, k=4 motifs
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()

    # For very small graphs (k <= 4), degree sequence + edge count is sufficient
    # For larger graphs, we'd need proper graph isomorphism
    return f"{num_nodes}_{num_edges}_{degrees}"


def find_frequent_motifs(
    G: nx.Graph,
    k: int = 3,
    min_frequency: int = 2,
    max_subgraphs: Optional[int] = 10000,
) -> Dict[str, Tuple[nx.Graph, int]]:
    """
    Find frequent k-node motifs in a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : int, default=3
        Motif size (number of nodes).
    min_frequency : int, default=2
        Minimum frequency to consider a motif as "frequent".
    max_subgraphs : int, optional
        Maximum number of subgraphs to extract.

    Returns
    -------
    Dict[str, Tuple[nx.Graph, int]]
        Dictionary mapping canonical motif form to (example_graph, frequency).
    """
    logger.info(f"Extracting {k}-node subgraphs from graph with {G.number_of_nodes()} nodes...")

    # Extract all k-node subgraphs
    subgraphs = extract_k_node_subgraphs(G, k=k, max_subgraphs=max_subgraphs)

    logger.info(f"Found {len(subgraphs)} {k}-node subgraphs")

    # Count frequencies by canonical form
    motif_counts: Counter[str] = Counter()
    motif_examples = {}

    for subgraph in subgraphs:
        canonical = canonical_motif_form(subgraph)
        motif_counts[canonical] += 1
        if canonical not in motif_examples:
            motif_examples[canonical] = subgraph

    # Filter by minimum frequency
    frequent_motifs = {
        canonical: (motif_examples[canonical], count)
        for canonical, count in motif_counts.items()
        if count >= min_frequency
    }

    logger.info(f"Found {len(frequent_motifs)} frequent motifs (frequency >= {min_frequency})")

    return frequent_motifs


def generate_null_model(
    G: nx.Graph,
    n_random: int = 100,
    preserve_degree: bool = True,
) -> List[nx.Graph]:
    """
    Generate random graphs (null models) with same properties as original graph.

    Parameters
    ----------
    G : nx.Graph
        Original graph.
    n_random : int, default=100
        Number of random graphs to generate.
    preserve_degree : bool, default=True
        If True, uses configuration model to preserve degree distribution.
        If False, uses Erdos-Renyi random graph.

    Returns
    -------
    List[nx.Graph]
        List of random graphs.
    """
    logger.info(f"Generating {n_random} null model graphs...")

    random_graphs = []

    for i in range(n_random):
        if preserve_degree:
            # Configuration model: preserves degree sequence
            degree_sequence = [G.degree(n) for n in G.nodes()]
            try:
                random_G = nx.configuration_model(degree_sequence, create_using=nx.Graph)
                # Remove self-loops and parallel edges
                random_G = nx.Graph(random_G)
                random_G.remove_edges_from(nx.selfloop_edges(random_G))
            except nx.NetworkXError:
                # Fallback to Erdos-Renyi if configuration model fails
                n = G.number_of_nodes()
                m = G.number_of_edges()
                p = 2 * m / (n * (n - 1))
                random_G = nx.erdos_renyi_graph(n, p)
        else:
            # Erdos-Renyi random graph
            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = 2 * m / (n * (n - 1))
            random_G = nx.erdos_renyi_graph(n, p)

        random_graphs.append(random_G)

    logger.info(f"Generated {len(random_graphs)} null model graphs")

    return random_graphs


def calculate_motif_significance(
    G: nx.Graph,
    motif: nx.Graph,
    null_models: List[nx.Graph],
    k: int = 3,
) -> Tuple[float, float, int]:
    """
    Calculate Z-score and p-value for a motif compared to null models.

    Parameters
    ----------
    G : nx.Graph
        Original graph.
    motif : nx.Graph
        Motif graph to test.
    null_models : List[nx.Graph]
        List of random graphs (null models).
    k : int, default=3
        Motif size.

    Returns
    -------
    Tuple[float, float, int]
        (z_score, p_value, observed_frequency)
    """
    # Count frequency in original graph
    canonical = canonical_motif_form(motif)
    subgraphs = extract_k_node_subgraphs(G, k=k, max_subgraphs=10000)
    observed = sum(1 for sg in subgraphs if canonical_motif_form(sg) == canonical)

    # Count frequencies in null models
    null_frequencies = []
    for null_G in null_models:
        null_subgraphs = extract_k_node_subgraphs(null_G, k=k, max_subgraphs=1000)
        null_count = sum(1 for sg in null_subgraphs if canonical_motif_form(sg) == canonical)
        null_frequencies.append(null_count)

    if len(null_frequencies) == 0:
        return 0.0, 1.0, observed

    # Calculate statistics
    mean_null = np.mean(null_frequencies)
    std_null = np.std(null_frequencies) + 1e-10  # Avoid division by zero

    z_score = (observed - mean_null) / std_null if std_null > 0 else 0.0

    # P-value: one-tailed test (motif is more frequent than expected)
    p_value = 1 - stats.norm.cdf(z_score) if std_null > 0 else 1.0

    return z_score, p_value, observed


def find_significant_motifs(
    G: nx.Graph,
    k: int = 3,
    min_frequency: int = 2,
    n_null_models: int = 50,
    z_threshold: float = 2.0,
    max_subgraphs: Optional[int] = 10000,
) -> List[MotifResult]:
    """
    Find statistically significant motifs in a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : int, default=3
        Motif size (number of nodes).
    min_frequency : int, default=2
        Minimum frequency to consider.
    n_null_models : int, default=50
        Number of null model graphs to generate.
    z_threshold : float, default=2.0
        Minimum Z-score for significance (default: 2.0 = ~p < 0.05).
    max_subgraphs : int, optional
        Maximum subgraphs to extract.

    Returns
    -------
    List[MotifResult]
        List of significant motifs with statistics.
    """
    logger.info(f"Finding significant {k}-node motifs...")

    # Find frequent motifs
    frequent_motifs = find_frequent_motifs(
        G, k=k, min_frequency=min_frequency, max_subgraphs=max_subgraphs
    )

    if len(frequent_motifs) == 0:
        logger.warning("No frequent motifs found.")
        return []

    # Generate null models
    null_models = generate_null_model(G, n_random=n_null_models, preserve_degree=True)

    # Calculate significance for each motif
    significant_motifs = []

    for canonical, (motif_graph, frequency) in frequent_motifs.items():
        z_score, p_value, observed = calculate_motif_significance(
            G, motif_graph, null_models, k=k
        )

        if z_score >= z_threshold:
            # Get node names from the motif
            nodes = list(motif_graph.nodes())
            significant_motifs.append(
                MotifResult(
                    motif_id=canonical,
                    nodes=nodes,
                    frequency=observed,
                    z_score=z_score,
                    p_value=p_value,
                )
            )

    # Sort by Z-score (most significant first)
    significant_motifs.sort(key=lambda x: x.z_score, reverse=True)

    logger.info(f"Found {len(significant_motifs)} significant motifs (Z >= {z_threshold})")

    return significant_motifs


def compare_motifs_pd_vs_control(
    G_pd: nx.Graph,
    G_control: nx.Graph,
    k: int = 3,
    min_frequency: int = 2,
    max_subgraphs: Optional[int] = 5000,
) -> pd.DataFrame:
    """
    Compare motif frequencies between PD and Control graphs.

    Parameters
    ----------
    G_pd : nx.Graph
        PD graph.
    G_control : nx.Graph
        Control graph.
    k : int, default=3
        Motif size.
    min_frequency : int, default=2
        Minimum frequency to consider.
    max_subgraphs : int, optional
        Maximum subgraphs to extract per graph.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: motif_id, pd_frequency, control_frequency,
        frequency_diff, fold_change.
    """
    logger.info("Comparing motifs between PD and Control graphs...")

    # Find motifs in both graphs
    motifs_pd = find_frequent_motifs(G_pd, k=k, min_frequency=min_frequency, max_subgraphs=max_subgraphs)
    motifs_control = find_frequent_motifs(G_control, k=k, min_frequency=min_frequency, max_subgraphs=max_subgraphs)

    # Get all unique motifs
    all_motifs = set(motifs_pd.keys()) | set(motifs_control.keys())

    # Build comparison DataFrame
    comparison_data = []
    for motif_id in all_motifs:
        pd_freq = motifs_pd.get(motif_id, (None, 0))[1]
        control_freq = motifs_control.get(motif_id, (None, 0))[1]

        freq_diff = pd_freq - control_freq
        fold_change = pd_freq / (control_freq + 1e-10)  # Avoid division by zero

        comparison_data.append({
            'motif_id': motif_id,
            'pd_frequency': pd_freq,
            'control_frequency': control_freq,
            'frequency_diff': freq_diff,
            'fold_change': fold_change,
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('frequency_diff', ascending=False)

    logger.info(f"Compared {len(df)} motifs between PD and Control")

    return df


def visualize_motif(
    motif: nx.Graph,
    node_names: Optional[Dict] = None,
    ax=None,
    title: Optional[str] = None,
) -> None:
    """
    Visualize a single motif.

    Parameters
    ----------
    motif : nx.Graph
        Motif graph to visualize.
    node_names : Dict, optional
        Mapping from node IDs to gene names.
        If None, uses node IDs directly.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Relabel nodes if names provided
    if node_names:
        mapping = {n: node_names.get(n, str(n)) for n in motif.nodes()}
        motif = nx.relabel_nodes(motif, mapping)

    # Layout
    pos = nx.spring_layout(motif, seed=42)

    # Draw
    nx.draw_networkx_nodes(motif, pos, ax=ax, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(motif, pos, ax=ax, edge_color='gray', width=2)
    nx.draw_networkx_labels(motif, pos, ax=ax, font_size=8)

    if title:
        ax.set_title(title, fontsize=12)

    ax.axis('off')


def visualize_top_motifs(
    motif_results: List[MotifResult],
    G: nx.Graph,
    node_names: Optional[List[str]] = None,
    top_n: int = 10,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Visualize top N significant motifs.

    Parameters
    ----------
    motif_results : List[MotifResult]
        List of motif results.
    G : nx.Graph
        Original graph (to extract actual subgraphs).
    node_names : List[str], optional
        List of node names matching graph node order.
    top_n : int, default=10
        Number of top motifs to visualize.
    figsize : Tuple[int, int], default=(15, 10)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    n_motifs = min(top_n, len(motif_results))
    n_cols = 5
    n_rows = (n_motifs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Create node name mapping if provided
    node_name_map = {}
    if node_names:
        graph_nodes = list(G.nodes())
        for i, node in enumerate(graph_nodes):
            if i < len(node_names):
                node_name_map[node] = node_names[i]

    for idx, motif_result in enumerate(motif_results[:n_motifs]):
        ax = axes[idx]

        # Find a subgraph matching this motif
        # For simplicity, use the nodes from the result
        if motif_result.nodes:
            try:
                subgraph = G.subgraph(motif_result.nodes)
                if subgraph.number_of_nodes() > 0:
                    title = f"Z={motif_result.z_score:.2f}\nFreq={motif_result.frequency}"
                    visualize_motif(subgraph, node_name_map, ax=ax, title=title)
            except:
                ax.text(0.5, 0.5, f"Motif {idx+1}\nZ={motif_result.z_score:.2f}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            ax.axis('off')

    # Hide unused axes
    for idx in range(n_motifs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig
