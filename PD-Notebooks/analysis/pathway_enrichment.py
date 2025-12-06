"""
Pathway enrichment analysis for gene clusters/modules.

This module provides functions for:
- KEGG pathway enrichment
- Reactome pathway enrichment
- Over-representation analysis (ORA) with hypergeometric test
- Jaccard index calculation for module-pathway similarity
- Visualization of enriched pathways

Supports multiple methods:
1. gprofiler (recommended) - API-based, supports KEGG, Reactome, GO
2. Enrichr - Web API for pathway enrichment
3. Custom ORA - Uses local pathway databases if available
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, fisher_exact
from statsmodels.stats.multitest import multipletests

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)

# Try to import pathway enrichment libraries
try:
    from gprofiler import GProfiler  # type: ignore
    GPROFILER_AVAILABLE = True
except ImportError:
    GPROFILER_AVAILABLE = False

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class PathwayEnrichmentResult:
    """
    Container for pathway enrichment results.

    Attributes
    ----------
    pathway_id : str
        Pathway identifier (e.g., "hsa05012" for KEGG).
    pathway_name : str
        Pathway name/description.
    source : str
        Database source ("KEGG", "Reactome", etc.).
    p_value : float
        Raw p-value from hypergeometric test.
    fdr : float
        False discovery rate (FDR) after multiple testing correction.
    overlap_count : int
        Number of genes in module that are in pathway.
    module_size : int
        Size of the gene module/cluster.
    pathway_size : int
        Total number of genes in pathway.
    background_size : int
        Total number of genes in background.
    overlap_genes : List[str]
        List of genes that overlap between module and pathway.
    enrichment_ratio : float
        Fold enrichment (observed/expected).
    jaccard_index : float
        Jaccard similarity between module and pathway.
    """
    pathway_id: str
    pathway_name: str
    source: str
    p_value: float
    fdr: float
    overlap_count: int
    module_size: int
    pathway_size: int
    background_size: int
    overlap_genes: List[str]
    enrichment_ratio: float
    jaccard_index: float


def calculate_jaccard_index(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard index between two gene sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set1 : Set[str]
        First gene set.
    set2 : Set[str]
        Second gene set.

    Returns
    -------
    float
        Jaccard index (0-1).
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def hypergeometric_enrichment(
    module_genes: List[str],
    pathway_genes: List[str],
    background_genes: List[str],
) -> Tuple[float, float, int, int, int, List[str]]:
    """
    Perform hypergeometric test for pathway enrichment.

    Tests if module_genes are enriched in pathway_genes compared to background.

    Parameters
    ----------
    module_genes : List[str]
        Genes in the module/cluster.
    pathway_genes : List[str]
        Genes in the pathway.
    background_genes : List[str]
        Background gene universe.

    Returns
    -------
    Tuple
        (p_value, enrichment_ratio, overlap_count, module_size, pathway_size, overlap_genes)
    """
    module_set = set(g.upper() for g in module_genes)
    pathway_set = set(g.upper() for g in pathway_genes)
    background_set = set(g.upper() for g in background_genes)

    # Filter to genes in background
    module_set = module_set & background_set
    pathway_set = pathway_set & background_set

    # Calculate overlap
    overlap = module_set & pathway_set
    overlap_count = len(overlap)
    module_size = len(module_set)
    pathway_size = len(pathway_set)
    background_size = len(background_set)

    if module_size == 0 or pathway_size == 0:
        return 1.0, 0.0, 0, module_size, pathway_size, []

    # Hypergeometric test
    # H0: genes are randomly distributed
    # H1: module genes are enriched in pathway
    # P(X >= overlap_count) where X ~ Hypergeometric(N, K, n)
    # N = background_size, K = pathway_size, n = module_size

    p_value = hypergeom.sf(
        overlap_count - 1,  # k-1 (we want P(X >= k))
        background_size,    # N
        pathway_size,       # K
        module_size         # n
    )

    # Expected overlap by chance
    expected = (pathway_size / background_size) * module_size
    enrichment_ratio = overlap_count / expected if expected > 0 else 0.0

    return (
        p_value,
        enrichment_ratio,
        overlap_count,
        module_size,
        pathway_size,
        sorted(list(overlap))
    )


def enrichr_enrichment(
    gene_list: List[str],
    gene_sets: List[str] = ["KEGG_2021_Human", "Reactome_2022"],
    background_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Perform pathway enrichment using Enrichr API.

    Enrichr is a web-based tool that provides pathway enrichment analysis.
    It's free and doesn't require installation, just the requests library.

    Parameters
    ----------
    gene_list : List[str]
        List of gene symbols to test.
    gene_sets : List[str]
        Enrichr gene set libraries to query.
        Options: "KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2021", etc.
    background_genes : List[str], optional
        Background gene universe (for custom analysis).
        Note: Enrichr uses its own background by default.

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns: pathway_name, p_value, fdr, overlap_count, etc.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library required for Enrichr API. Install with: pip install requests")

    ENRICHR_ADD_URL = "https://maayanlab.cloud/Enrichr/addList"
    ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"

    genes_str = "\n".join(gene_list)

    # Add gene list to Enrichr
    try:
        response = requests.post(
            ENRICHR_ADD_URL,
            files={"list": (None, genes_str)},
            data={"description": "PD_cluster"}
        )

        if not response.ok:
            raise Exception(f"Enrichr API error: {response.status_code}")

        data = response.json()
        user_list_id = data.get("userListId")

        # Query each gene set library
        all_results = []

        for gene_set in gene_sets:
            query_response = requests.get(
                ENRICHR_ENRICH_URL,
                params={
                    "userListId": user_list_id,
                    "backgroundType": gene_set
                }
            )

            if query_response.ok:
                results = query_response.json()
                if gene_set in results:
                    for item in results[gene_set]:
                        all_results.append({
                            "pathway_id": item[0],  # Pathway ID
                            "pathway_name": item[1],  # Pathway name
                            "p_value": item[2],
                            "fdr": item[3],  # Adjusted p-value
                            "overlap_count": len(item[5]),  # Number of overlapping genes
                            "overlap_genes": item[5],  # List of overlapping genes
                            "source": gene_set,
                            "enrichment_ratio": item[4] if len(item) > 4 else None,  # Z-score or odds ratio
                        })

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df = df.sort_values("fdr")

        return df

    except Exception as e:
        logger.error(f"Enrichr API error: {e}")
        raise


def gprofiler_enrichment(
    gene_list: List[str],
    sources: List[str] = ["KEGG", "REAC"],
    background_genes: Optional[List[str]] = None,
    organism: str = "hsapiens",
) -> pd.DataFrame:
    """
    Perform pathway enrichment using g:Profiler API.

    g:Profiler is a comprehensive functional enrichment tool that queries:
    - KEGG pathways
    - Reactome pathways
    - Gene Ontology (GO) terms
    - And many other databases

    It's the recommended method as it's fast, reliable, and supports custom backgrounds.

    Parameters
    ----------
    gene_list : List[str]
        List of gene symbols to test.
    sources : List[str]
        Data sources: "KEGG", "REAC" (Reactome), "GO:BP", "GO:MF", "GO:CC", etc.
    background_genes : List[str], optional
        Background gene universe (all genes in your network).
        If None, uses all genes in g:Profiler database.
    organism : str, default="hsapiens"
        Organism code.

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns: pathway_id, pathway_name, p_value, fdr, etc.
    """
    if not GPROFILER_AVAILABLE:
        raise ImportError(
            "gprofiler-official required. Install with: pip install gprofiler-official\n"
            "g:Profiler is a tool that queries KEGG, Reactome, GO, and other databases via API."
        )

    gp_instance = GProfiler(return_dataframe=True)

    # Run enrichment
    try:
        result = gp_instance.profile(
            query=gene_list,
            organism=organism,
            sources=sources,
            background=background_genes,  # Custom background if provided
            significance_threshold_method="fdr",
            user_threshold=0.05,
        )

        if result is None or len(result) == 0:
            return pd.DataFrame()

        # Rename columns for consistency
        column_mapping = {
            "native": "pathway_id",
            "name": "pathway_name",
            "p_value": "p_value",
            "adjusted_p_value": "fdr",
            "intersection_size": "overlap_count",
            "intersection": "overlap_genes",
            "source": "source",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in result.columns:
                result = result.rename(columns={old_col: new_col})

        # Calculate enrichment ratio if not present
        if "enrichment_ratio" not in result.columns:
            if "query_size" in result.columns and "term_size" in result.columns:
                # enrichment = (overlap / query_size) / (term_size / background_size)
                background_size = len(background_genes) if background_genes else 20000
                result["enrichment_ratio"] = (
                    result["overlap_count"] / result["query_size"]
                ) / (result["term_size"] / background_size)
            else:
                result["enrichment_ratio"] = np.nan

        return result

    except Exception as e:
        logger.error(f"g:Profiler API error: {e}")
        raise


def enrich_clusters_pathways(
    cluster_results,
    node_names: List[str],
    background_genes: Optional[List[str]] = None,
    method: str = "gprofiler",
    sources: List[str] = ["KEGG", "REAC"],
    fdr_threshold: float = 0.05,
    min_overlap: int = 3,
) -> Dict[int, pd.DataFrame]:
    """
    Perform pathway enrichment for all clusters.

    Parameters
    ----------
    cluster_results
        ClusterResults object with cluster labels.
    node_names : List[str]
        List of gene names matching cluster labels.
    background_genes : List[str], optional
        Background gene universe. If None, uses all node_names.
    method : str, default="gprofiler"
        Enrichment method: "gprofiler", "enrichr", or "custom".
    sources : List[str], default=["KEGG", "REAC"]
        Data sources for gprofiler.
    fdr_threshold : float, default=0.05
        FDR threshold for significance.
    min_overlap : int, default=3
        Minimum number of overlapping genes required.

    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping cluster_id to enrichment results DataFrame.
    """
    if background_genes is None:
        background_genes = node_names

    enrichment_results = {}
    unique_clusters = np.unique(cluster_results.labels)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_results.labels == cluster_id)[0]
        module_genes = [node_names[i] for i in cluster_indices]

        if len(module_genes) <= 1:
            continue

        try:
            if method == "gprofiler" and GPROFILER_AVAILABLE:
                df = gprofiler_enrichment(
                    gene_list=module_genes,
                    sources=sources,
                    background_genes=background_genes,
                )
            elif method == "enrichr" and REQUESTS_AVAILABLE:
                # Map sources to Enrichr gene set names
                enrichr_sets = []
                for source in sources:
                    if source == "KEGG":
                        enrichr_sets.append("KEGG_2021_Human")
                    elif source == "REAC" or source == "Reactome":
                        enrichr_sets.append("Reactome_2022")
                    else:
                        enrichr_sets.append(f"{source}_2021")

                df = enrichr_enrichment(
                    gene_list=module_genes,
                    gene_sets=enrichr_sets,
                    background_genes=background_genes,
                )
            else:
                if method == "gprofiler" and not GPROFILER_AVAILABLE:
                    if REQUESTS_AVAILABLE:
                        method = "enrichr"
                        enrichr_sets = ["KEGG_2021_Human", "Reactome_2022"]
                        df = enrichr_enrichment(
                            gene_list=module_genes,
                            gene_sets=enrichr_sets,
                        )
                    else:
                        continue
                else:
                    continue

            # Filter by minimum overlap and FDR
            if len(df) > 0:
                df = df[df['overlap_count'] >= min_overlap]
                if 'fdr' in df.columns:
                    df = df[df['fdr'] <= fdr_threshold]
                df = df.sort_values('fdr' if 'fdr' in df.columns else 'p_value')

                enrichment_results[cluster_id] = df

        except Exception:
            continue

    return enrichment_results


def visualize_pathway_enrichment(
    enrichment_results: Dict[int, pd.DataFrame],
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Visualize pathway enrichment results as bar plots.

    Parameters
    ----------
    enrichment_results : Dict[int, pd.DataFrame]
        Enrichment results per cluster.
    top_n : int, default=10
        Number of top pathways to show per cluster.
    figsize : Tuple[int, int], default=(14, 6)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if len(enrichment_results) == 0:
        return None

    n_clusters = len(enrichment_results)
    n_cols = min(2, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (cluster_id, df) in enumerate(sorted(enrichment_results.items())):
        if len(df) == 0:
            continue

        ax = axes[idx]

        # Get top pathways
        top_pathways = df.head(top_n)

        # Create bar plot with -log10(FDR)
        if 'fdr' in top_pathways.columns:
            y_values = -np.log10(top_pathways['fdr'].values + 1e-10)  # Add small value to avoid log(0)
            y_label = "-log10(FDR)"
        else:
            y_values = -np.log10(top_pathways['p_value'].values + 1e-10)
            y_label = "-log10(p-value)"

        pathway_names = top_pathways['pathway_name'].values
        # Truncate long pathway names
        pathway_names = [name[:50] + "..." if len(name) > 50 else name for name in pathway_names]

        colors = plt.cm.viridis(y_values / y_values.max() if y_values.max() > 0 else 1)

        bars = ax.barh(range(len(pathway_names)), y_values, color=colors)
        ax.set_yticks(range(len(pathway_names)))
        ax.set_yticklabels(pathway_names, fontsize=9)
        ax.set_xlabel(y_label, fontsize=10)
        ax.set_title(f"Cluster {cluster_id} - Top {len(top_pathways)} Enriched Pathways", fontsize=11, fontweight="bold")
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, y_values)):
            ax.text(val + 0.1, i, f"{val:.2f}", va='center', fontsize=8)

    # Hide unused subplots
    for idx in range(len(enrichment_results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def calculate_pathway_jaccard(
    module_genes: List[str],
    pathway_genes: List[str],
) -> float:
    """
    Calculate Jaccard index between module and pathway gene sets.

    J(Module, Pathway) = |Module ∩ Pathway| / |Module ∪ Pathway|

    Parameters
    ----------
    module_genes : List[str]
        Genes in the module/cluster.
    pathway_genes : List[str]
        Genes in the pathway.

    Returns
    -------
    float
        Jaccard index (0-1).
    """
    module_set = set(g.upper() for g in module_genes)
    pathway_set = set(g.upper() for g in pathway_genes)
    return calculate_jaccard_index(module_set, pathway_set)


# Alias for backward compatibility
calculate_module_pathway_jaccard = calculate_pathway_jaccard
