"""
Graph mining and embedding analysis for PD gene-gene networks.

This module provides:
- Embedding reduction (UMAP/t-SNE)
- Clustering (KMeans, Leiden)
- Cluster visualization
- Cluster statistics
- PD gene validation
"""

from .embedding_analysis import (
    reduce_embeddings,
    cluster_embeddings,
    visualize_clusters,
    analyze_clusters,
    embedding_analysis_pipeline,
)

from .pd_validation import (
    PD_KNOWN_GENES,
    PDGeneValidation,
    validate_known_pd_genes,
    print_validation_summary,
    create_validation_dataframe,
    diagnose_pd_gene_presence,
)

from .motif_finding import (
    extract_k_node_subgraphs,
    find_frequent_motifs,
    find_significant_motifs,
    compare_motifs_pd_vs_control,
    visualize_motif,
    visualize_top_motifs,
    MotifResult,
)

__all__ = [
    "reduce_embeddings",
    "cluster_embeddings",
    "visualize_clusters",
    "analyze_clusters",
    "embedding_analysis_pipeline",
    "PD_KNOWN_GENES",
    "PDGeneValidation",
    "validate_known_pd_genes",
    "print_validation_summary",
    "create_validation_dataframe",
    "diagnose_pd_gene_presence",
    "extract_k_node_subgraphs",
    "find_frequent_motifs",
    "find_significant_motifs",
    "compare_motifs_pd_vs_control",
    "visualize_motif",
    "visualize_top_motifs",
    "MotifResult",
]
