"""
PD validation module for comparing findings against established PD knowledge.

This module provides validation functions to check:
- Overlap with known PD-associated genes
- Pathway enrichment validation
- Differential expression validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


# Known PD-associated genes
# Source: MDSGene, DisGeNET, literature review
PD_KNOWN_GENES = {
    # Monogenic PD genes (highly validated)
    'SNCA',      # Alpha-synuclein - first PD gene discovered
    'PRKN',      # Parkin - most common monogenic PD
    'PINK1',     # PTEN-induced kinase 1
    'DJ1',       # Protein deglycase DJ-1
    'LRRK2',     # Leucine-rich repeat kinase 2
    'VPS35',     # Vacuolar protein sorting 35
    'GBA',       # Glucosylceramidase beta
    'UCHL1',     # Ubiquitin carboxyl-terminal esterase L1
    'HTRA2',     # Serine protease HTRA2
    'ATP13A2',   # Lysosomal type 5 P-type ATPase
    'FBXO7',     # F-box protein 7
    'DNAJC6',    # DnaJ heat shock protein family member C6
    'SYNJ1',     # Synaptojanin 1

    # Risk factor genes
    'GBA1',      # Glucosylceramidase beta (alternative name)
    'MAPT',      # Microtubule-associated protein tau
    'GIGYF2',    # GRB10 interacting GYF protein 2
    'PLA2G6',    # Phospholipase A2 group VI
    'VPS13C',    # Vacuolar protein sorting 13 homolog C

    # Mitochondrial genes related to PD
    'POLG',      # DNA polymerase gamma
    'TFAM',      # Mitochondrial transcription factor A
    'NDUFAF2',   # NADH:ubiquinone oxidoreductase complex assembly factor 2
    'COX4I1',    # Cytochrome c oxidase subunit 4I1

    # Additional PD-related genes from recent studies
    'TMEM230',   # Transmembrane protein 230
    'CHCHD2',    # Coiled-coil-helix-coiled-coil-helix domain containing 2
    'PARK7',     # Parkinsonism associated deglycase (DJ1 alias)
    'PARK2',     # Parkin RBR E3 ubiquitin protein ligase (PRKN alias)
    'PARK6',     # PTEN induced putative kinase 1 (PINK1 alias)
}


@dataclass
class PDGeneValidation:
    """
    Container for PD gene validation results.

    Attributes
    ----------
    cluster_id : int
        Cluster identifier.
    found_genes : List[str]
        List of known PD genes found in this cluster.
    overlap_count : int
        Number of known PD genes found.
    cluster_size : int
        Total number of genes in cluster.
    overlap_percentage : float
        Percentage of known PD genes found (relative to total known PD genes).
    cluster_enrichment : float
        Percentage of cluster that consists of known PD genes.
    """

    cluster_id: int
    found_genes: List[str]
    overlap_count: int
    cluster_size: int
    overlap_percentage: float
    cluster_enrichment: float


def validate_known_pd_genes(
    cluster_labels: np.ndarray,
    node_names: List[str],
    gene_metadata: pd.DataFrame,
    gene_id_column: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
) -> Dict[int, PDGeneValidation]:
    """
    Validate clusters against known PD-associated genes.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels for each node (same length as node_names).
    node_names : List[str]
        List of gene IDs (Ensembl IDs or gene symbols).
    gene_metadata : pd.DataFrame
        Gene metadata DataFrame with gene annotations.
        Expected: index contains gene IDs matching node_names.
    gene_id_column : str, optional
        Column name in gene_metadata that matches node_names.
        If None, assumes node_names match gene_metadata index.
    gene_symbol_column : str, optional
        Column name in gene_metadata containing gene symbols.
        If None, will try to find 'Symbol', 'GeneName', or 'Gene symbol'.

    Returns
    -------
    Dict[int, PDGeneValidation]
        Dictionary mapping cluster ID to validation results.
    """
    # Find gene symbol column
    if gene_symbol_column is None:
        possible_columns = ['Symbol', 'GeneName', 'Gene symbol', 'gene_symbol', 'gene_name']
        for col in possible_columns:
            if col in gene_metadata.columns:
                gene_symbol_column = col
                break

        if gene_symbol_column is None:
            gene_symbol_column = None

    # Create mapping from gene IDs to gene symbols
    gene_id_to_symbol = {}

    for node_name in node_names:
        # Try to find this gene in metadata
        if node_name in gene_metadata.index:
            # Gene found in metadata index
            if gene_symbol_column and gene_symbol_column in gene_metadata.columns:
                gene_symbol = gene_metadata.loc[node_name, gene_symbol_column]
                if pd.notna(gene_symbol) and gene_symbol:
                    # Handle multiple symbols (split by comma/semicolon if needed)
                    symbol = str(gene_symbol).split(';')[0].split(',')[0].strip()
                    gene_id_to_symbol[node_name] = symbol.upper()
                else:
                    gene_id_to_symbol[node_name] = str(node_name).upper()
            else:
                # No symbol column, use ID
                gene_id_to_symbol[node_name] = str(node_name).upper()
        elif gene_id_column and gene_id_column in gene_metadata.columns:
            # Try to find by column
            matches = gene_metadata[gene_metadata[gene_id_column] == node_name]
            if len(matches) > 0:
                if gene_symbol_column and gene_symbol_column in gene_metadata.columns:
                    gene_symbol = matches.iloc[0][gene_symbol_column]
                    if pd.notna(gene_symbol) and gene_symbol:
                        symbol = str(gene_symbol).split(';')[0].split(',')[0].strip()
                        gene_id_to_symbol[node_name] = symbol.upper()
                    else:
                        gene_id_to_symbol[node_name] = str(node_name).upper()
                else:
                    gene_id_to_symbol[node_name] = str(node_name).upper()
            else:
                gene_id_to_symbol[node_name] = str(node_name).upper()
        else:
            # Not found, use ID directly
            gene_id_to_symbol[node_name] = str(node_name).upper()

    validation_results = {}

    unique_clusters = np.unique(cluster_labels)

    total_pd_genes_found = set()

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_gene_ids = [node_names[i] for i in cluster_indices]
        cluster_symbols = [
            gene_id_to_symbol.get(gene_id, gene_id).upper()
            for gene_id in cluster_gene_ids
        ]

        # Find overlap with known PD genes
        overlap = set(cluster_symbols) & PD_KNOWN_GENES
        total_pd_genes_found.update(overlap)

        overlap_count = len(overlap)
        cluster_size = len(cluster_symbols)

        # Calculate percentages
        overlap_percentage = (overlap_count / len(PD_KNOWN_GENES) * 100) if PD_KNOWN_GENES else 0
        cluster_enrichment = (overlap_count / cluster_size * 100) if cluster_size > 0 else 0

        validation_results[cluster_id] = PDGeneValidation(
            cluster_id=int(cluster_id),
            found_genes=sorted(list(overlap)),
            overlap_count=overlap_count,
            cluster_size=cluster_size,
            overlap_percentage=overlap_percentage,
            cluster_enrichment=cluster_enrichment,
        )

    return validation_results


def print_validation_summary(validation_results: Dict[int, PDGeneValidation]) -> None:
    """
    Print a formatted summary of validation results.

    Parameters
    ----------
    validation_results : Dict[int, PDGeneValidation]
        Validation results from validate_known_pd_genes.
    """
    print("\n" + "=" * 60)
    print("PD GENE VALIDATION SUMMARY")
    print("=" * 60)

    # Summary statistics
    total_pd_genes_found = set()
    clusters_with_pd_genes = 0

    for cluster_id, result in validation_results.items():
        if result.overlap_count > 0:
            clusters_with_pd_genes += 1
            total_pd_genes_found.update(result.found_genes)

    print(f"\nTotal known PD genes in database: {len(PD_KNOWN_GENES)}")
    print(f"Unique PD genes found: {len(total_pd_genes_found)} ({len(total_pd_genes_found)/len(PD_KNOWN_GENES)*100:.1f}%)")
    print(f"Clusters containing PD genes: {clusters_with_pd_genes}/{len(validation_results)}")

    print(f"\nPD genes found: {', '.join(sorted(total_pd_genes_found))}")

    # Per-cluster details
    print("\n" + "-" * 60)
    print("PER-CLUSTER RESULTS")
    print("-" * 60)

    for cluster_id in sorted(validation_results.keys()):
        result = validation_results[cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  Cluster size: {result.cluster_size} genes")
        print(f"  PD genes found: {result.overlap_count}")

        if result.found_genes:
            print(f"  Genes: {', '.join(result.found_genes)}")
            print(f"  Cluster enrichment: {result.cluster_enrichment:.2f}%")
        else:
            print(f"  No known PD genes found in this cluster")

    print("\n" + "=" * 60)


def create_validation_dataframe(
    validation_results: Dict[int, PDGeneValidation]
) -> pd.DataFrame:
    """
    Convert validation results to a pandas DataFrame for easy analysis.

    Parameters
    ----------
    validation_results : Dict[int, PDGeneValidation]
        Validation results from validate_known_pd_genes.

    Returns
    -------
    pd.DataFrame
        DataFrame with validation results.
    """
    rows = []
    for cluster_id, result in validation_results.items():
        rows.append({
            'cluster_id': result.cluster_id,
            'cluster_size': result.cluster_size,
            'pd_genes_found': result.overlap_count,
            'pd_genes': ', '.join(result.found_genes) if result.found_genes else 'None',
            'overlap_percentage': result.overlap_percentage,
            'cluster_enrichment': result.cluster_enrichment,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('pd_genes_found', ascending=False)

    return df


def diagnose_pd_gene_presence(
    node_names: List[str],
    gene_metadata: pd.DataFrame,
    original_expression: Optional[pd.DataFrame] = None,
    processed_expression: Optional[pd.DataFrame] = None,
    gene_symbol_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Diagnose which known PD genes are present in the dataset.

    Checks:
    1. Which PD genes are in the current dataset (node_names)
    2. Which PD genes might have been filtered out during preprocessing
    3. Gene symbol mapping status

    Parameters
    ----------
    node_names : List[str]
        List of gene IDs in current analysis (e.g., after HVG selection).
    gene_metadata : pd.DataFrame
        Gene metadata with gene annotations.
    original_expression : pd.DataFrame, optional
        Original expression matrix before preprocessing.
    processed_expression : pd.DataFrame, optional
        Processed expression matrix (after HVG selection).
    gene_symbol_column : str, optional
        Column name containing gene symbols.

    Returns
    -------
    pd.DataFrame
        Diagnostic DataFrame showing status of each known PD gene.
    """
    # Find gene symbol column
    if gene_symbol_column is None:
        possible_columns = ['Symbol', 'GeneName', 'Gene symbol', 'gene_symbol', 'gene_name']
        for col in possible_columns:
            if col in gene_metadata.columns:
                gene_symbol_column = col
                break

    # Build mapping: gene_symbol -> gene_id(s)
    symbol_to_ids: Dict[str, List[str]] = {}
    all_gene_ids = set()

    if gene_symbol_column and gene_symbol_column in gene_metadata.columns:
        for gene_id in gene_metadata.index:
            all_gene_ids.add(gene_id)
            symbol = gene_metadata.loc[gene_id, gene_symbol_column]
            if pd.notna(symbol) and symbol:
                symbol_clean = str(symbol).split(';')[0].split(',')[0].strip().upper()
                if symbol_clean not in symbol_to_ids:
                    symbol_to_ids[symbol_clean] = []
                symbol_to_ids[symbol_clean].append(gene_id)

    # Check status of each known PD gene
    diagnostic_rows = []

    for pd_gene in sorted(PD_KNOWN_GENES):
        pd_gene_upper = pd_gene.upper()

        # Check if gene symbol exists in metadata
        has_symbol_match = pd_gene_upper in symbol_to_ids

        if has_symbol_match:
            matching_ids = symbol_to_ids[pd_gene_upper]

            # Check which IDs are in current analysis
            in_current = [gid for gid in matching_ids if gid in node_names]

            # Check which IDs were in original
            in_original: Optional[List[str]] = None
            if original_expression is not None:
                in_original = [gid for gid in matching_ids if gid in original_expression.index]

            # Status
            if len(in_current) > 0:
                status = "PRESENT in current analysis"
            elif original_expression is not None and in_original is not None and len(in_original) > 0:
                status = "FILTERED OUT (was in original)"
            elif len(matching_ids) > 0:
                status = "NOT IN CURRENT ANALYSIS"
            else:
                status = "NOT FOUND in metadata"

            diagnostic_rows.append({
                'pd_gene': pd_gene,
                'status': status,
                'in_current_analysis': len(in_current) > 0,
                'in_original': len(in_original) if in_original is not None else 'N/A',
                'matching_ids_count': len(matching_ids),
                'example_id': matching_ids[0] if matching_ids else 'N/A',
                'current_ids': ', '.join(in_current[:3]) if in_current else 'None',
            })
        else:
            # Try to find by partial match or alias
            partial_matches = [s for s in symbol_to_ids.keys() if pd_gene_upper in s or s in pd_gene_upper]

            diagnostic_rows.append({
                'pd_gene': pd_gene,
                'status': f"NO SYMBOL MATCH (found {len(partial_matches)} partial matches)" if partial_matches else "NOT FOUND",
                'in_current_analysis': False,
                'in_original': 'N/A',
                'matching_ids_count': 0,
                'example_id': 'N/A',
                'current_ids': 'None',
            })

    return pd.DataFrame(diagnostic_rows)
