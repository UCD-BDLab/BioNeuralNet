"""
Multi-omics data loaders for Parkinson's disease brain tissue datasets.

This module provides loaders for:
- GEO Series Matrix files (RNA expression from microarray/RNA-seq)
- Proteomics CSV files (protein abundance)
- Multi-omics integration at the gene level
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, cast
import re

import pandas as pd
import numpy as np

import sys
from pathlib import Path as PathLib

project_root = PathLib(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class OmicsData:
    """
    Container for single-omic data.

    Attributes
    ----------
    expression : pd.DataFrame
        Expression/abundance matrix (features × samples).
        Index: feature IDs (probe IDs, gene symbols, protein IDs, etc.)
        Columns: sample identifiers
    sample_metadata : pd.DataFrame
        Sample-level metadata.
        Index: sample identifiers (matching expression columns)
        Columns: condition, tissue, etc.
    feature_metadata : pd.DataFrame
        Feature-level metadata (gene symbols, descriptions, etc.)
        Index: feature IDs (matching expression index)
    omic_type : str
        Type of omic: "rna", "proteomics", "meth", etc.
    """
    expression: pd.DataFrame
    sample_metadata: pd.DataFrame
    feature_metadata: pd.DataFrame
    omic_type: str


@dataclass
class MultiOmicsData:
    """
    Container for multi-omics data integrated at gene level.

    Attributes
    ----------
    rna : Optional[OmicsData]
        RNA expression data
    proteomics : Optional[OmicsData]
        Proteomics data
    meth : Optional[OmicsData]
        Methylation data (for future use)
    common_genes : List[str]
        List of gene symbols common across all omics
    """
    rna: Optional[OmicsData] = None
    proteomics: Optional[OmicsData] = None
    meth: Optional[OmicsData] = None
    common_genes: Optional[List[str]] = None


def parse_geo_series_matrix(
    filepath: Union[str, Path],
    data_start_marker: str = "!series_matrix_table_begin",
    extract_platform_id: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Parse a GEO Series Matrix file and extract expression data and metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to the GEO Series Matrix file (.txt)
    data_start_marker : str, default="!series_matrix_table_begin"
        Marker line indicating start of data table

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]
        (expression_matrix, sample_metadata, platform_id)
        - expression_matrix: features × samples (index: probe IDs, columns: sample IDs)
        - sample_metadata: sample-level metadata extracted from !Sample_* lines
        - platform_id: GEO platform ID (e.g., "GPL570") if extract_platform_id=True
    """
    filepath = Path(filepath)
    logger.info(f"Parsing GEO Series Matrix file: {filepath}")

    # Read file and find data start
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find data table start
    data_start_idx = None
    for i, line in enumerate(lines):
        if data_start_marker in line:
            data_start_idx = i + 1  # Next line is header
            break

    if data_start_idx is None:
        raise ValueError(f"Could not find data table start marker '{data_start_marker}' in file.")

    # Parse metadata lines (before data table)
    metadata_lines: Dict[str, Union[List[str], List[List[str]]]] = {}
    platform_id: Optional[str] = None
    for i in range(data_start_idx):
        line = lines[i].strip()
        if line.startswith('!'):
            parts = line.split('\t', 1)
            if len(parts) == 2:
                key = parts[0]
                metadata_values: List[str] = [v.strip('"') for v in parts[1].split('\t')]

                # Handle multiple lines with same key (e.g., multiple characteristics_ch1)
                if key in metadata_lines:
                    # Convert to list of lists if not already
                    existing_value = metadata_lines[key]
                    if isinstance(existing_value, list) and len(existing_value) > 0:
                        # Check if it's a List[str] (single list) that needs to be converted to List[List[str]]
                        if not isinstance(existing_value[0], list):
                            # Convert List[str] to List[List[str]]
                            existing_value_str = cast(List[str], existing_value)
                            metadata_lines[key] = [existing_value_str]
                        # Now it's guaranteed to be List[List[str]], so we can append
                        current_value = metadata_lines[key]
                        if isinstance(current_value, list) and len(current_value) > 0:
                            if isinstance(current_value[0], list):
                                # Type narrowing: current_value is List[List[str]]
                                cast(List[List[str]], current_value).append(metadata_values)
                else:
                    metadata_lines[key] = metadata_values

                # Extract platform ID if requested
                if extract_platform_id and 'platform_id' in key.lower():
                    platform_id = metadata_values[0] if metadata_values else None

    # Parse data table
    # First line after marker is header
    header_line = lines[data_start_idx].strip()
    header_parts = header_line.split('\t')
    # Remove quotes
    sample_ids = [col.strip('"') for col in header_parts[1:]]  # Skip ID_REF column

    # Read data table
    data_lines: List[List[float]] = []
    feature_ids: List[str] = []
    for i in range(data_start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('!'):
            break
        parts = line.split('\t')
        if len(parts) > 1:
            feature_id = parts[0].strip('"')
            # Handle null, empty, and invalid values
            values: List[float] = []
            for v in parts[1:]:
                v_clean = v.strip('"').strip().lower()
                if v_clean in ['', 'null', 'na', 'n/a', 'nan', 'none']:
                    values.append(float('nan'))
                else:
                    try:
                        values.append(float(v_clean))
                    except (ValueError, TypeError):
                        values.append(float('nan'))
            feature_ids.append(feature_id)
            data_lines.append(values)

    # Create expression matrix
    expression_df = pd.DataFrame(
        data_lines,
        index=feature_ids,
        columns=sample_ids
    )

    # Build sample metadata from !Sample_* lines
    sample_metadata_dict: Dict[str, List[str]] = {}
    characteristics_lines: List[List[str]] = []  # Collect all characteristics lines

    for key, metadata_values_raw in metadata_lines.items():
        metadata_values: Union[List[str], List[List[str]]] = metadata_values_raw
        if key.startswith('!Sample_'):
            # Extract field name (e.g., !Sample_geo_accession -> geo_accession)
            field_name = key.replace('!Sample_', '').replace('!', '')

            # Handle characteristics_ch1 specially (can be multiple lines)
            if field_name == 'characteristics_ch1':
                # Check if metadata_values is a list of lists (multiple characteristics lines)
                if isinstance(metadata_values, list) and len(metadata_values) > 0:
                    if isinstance(metadata_values[0], list):
                        # Already a list of lists
                        characteristics_lines.extend(cast(List[List[str]], metadata_values))
                    else:
                        # Single characteristics line - needs to be List[str]
                        characteristics_lines.append(cast(List[str], metadata_values))
            elif isinstance(metadata_values, list) and len(metadata_values) == len(sample_ids):
                # Regular field with correct length - must be List[str]
                if not (len(metadata_values) > 0 and isinstance(metadata_values[0], list)):
                    # Type narrowing: it's List[str], not List[List[str]]
                    sample_metadata_dict[field_name] = cast(List[str], metadata_values)
            elif isinstance(metadata_values, list) and len(metadata_values) > 0:
                if isinstance(metadata_values[0], list) and len(metadata_values[0]) == len(sample_ids):
                    # List of lists - use first
                    sample_metadata_dict[field_name] = cast(List[List[str]], metadata_values)[0]

    # Add condition labels if available
    condition_labels = ['UNKNOWN'] * len(sample_ids)

    # Process characteristics lines (can be multiple lines)
    if characteristics_lines:
        for char_line in characteristics_lines:
            for i, char in enumerate(char_line):
                if i < len(condition_labels) and char:
                    char_lower = str(char).lower()
                    # Look for disease state information
                    if 'disease' in char_lower or 'condition' in char_lower:
                        # Extract condition (handle "Parkinson's disease", "control", etc.)
                        if 'parkinson' in char_lower or 'pd' in char_lower:
                            condition_labels[i] = 'PD'
                        elif 'control' in char_lower or 'ctrl' in char_lower or 'normal' in char_lower:
                            condition_labels[i] = 'CONTROL'
                        else:
                            # Try to extract after colon (e.g., "disease state: Parkinson's disease")
                            match = re.search(r':\s*([^:]+)', char, re.IGNORECASE)
                            if match:
                                label = match.group(1).strip()
                                label_lower = label.lower()
                                if 'parkinson' in label_lower or 'pd' in label_lower:
                                    condition_labels[i] = 'PD'
                                elif 'control' in label_lower or 'ctrl' in label_lower or 'normal' in label_lower:
                                    condition_labels[i] = 'CONTROL'
                                else:
                                    # Use the extracted label as-is (capitalize)
                                    condition_labels[i] = label.upper()

    # Fallback: try to infer from sample titles if still unknown
    if 'title' in sample_metadata_dict:
        for i, title in enumerate(sample_metadata_dict['title']):
            if i < len(condition_labels) and condition_labels[i] == 'UNKNOWN':
                title_upper = str(title).upper()
                if 'PD' in title_upper or 'PARKINSON' in title_upper:
                    condition_labels[i] = 'PD'
                elif title_upper.startswith('C-') or 'CONTROL' in title_upper or 'CTRL' in title_upper:
                    condition_labels[i] = 'CONTROL'

    sample_metadata = pd.DataFrame(
        {
            'condition': condition_labels,
            **{k: v for k, v in sample_metadata_dict.items() if k != 'characteristics_ch1'}
        },
        index=sample_ids
    )

    logger.info(
        f"Parsed GEO file: {expression_df.shape[0]} features, "
        f"{expression_df.shape[1]} samples"
    )
    if platform_id:
        logger.info(f"Detected platform ID: {platform_id}")

    return expression_df, sample_metadata, platform_id


def map_probe_to_gene_symbol(
    probe_ids: List[str],
    platform_id: Optional[str] = None,
    annotation_file: Optional[Union[str, Path]] = None,
    annotation_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Map Affymetrix probe IDs to gene symbols.

    Attempts to load platform annotation files if available, otherwise uses placeholder.

    Parameters
    ----------
    probe_ids : List[str]
        List of probe IDs to map
    platform_id : str, optional
        GEO platform ID (e.g., "GPL570")
    annotation_file : str or Path, optional
        Path to annotation file (if available)
    annotation_dir : str or Path, optional
        Directory to search for annotation files (looks for GPL*.annot or GPL*.txt)

    Returns
    -------
    pd.DataFrame
        Feature metadata with columns: probe_id, gene_symbol
        Index: probe_ids
    """
    logger.info(f"Mapping {len(probe_ids)} probe IDs to gene symbols...")

    # Try to load annotation file
    annotation_df = None

    if annotation_file:
        annotation_file = Path(annotation_file)
        if annotation_file.exists():
            try:
                annotation_df = _load_geo_annotation_file(annotation_file)
                logger.info(f"Loaded annotation file: {annotation_file}")
            except Exception as e:
                logger.warning(f"Failed to load annotation file {annotation_file}: {e}")

    # Try to find annotation file in directory
    if annotation_df is None and platform_id and annotation_dir:
        annotation_dir = Path(annotation_dir)
        # Look for common annotation file patterns
        import glob
        patterns = [
            f"{platform_id}.annot",
            f"{platform_id}.txt",
            f"{platform_id}_annot.txt",
            f"{platform_id}_full_table.txt",
        ]
        # Try exact matches first
        for pattern in patterns:
            annot_path = annotation_dir / pattern
            if annot_path.exists():
                try:
                    annotation_df = _load_geo_annotation_file(annot_path)
                    logger.info(f"Found and loaded annotation file: {annot_path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load {annot_path}: {e}")

        # If not found, try glob pattern for files like GPL570-55999.txt
        if annotation_df is None:
            glob_pattern = str(annotation_dir / f"{platform_id}-*.txt")
            matches = glob.glob(glob_pattern)
            if matches:
                # Use the first match
                annot_path = Path(matches[0])
                try:
                    annotation_df = _load_geo_annotation_file(annot_path)
                    logger.info(f"Found and loaded annotation file: {annot_path}")
                except Exception as e:
                    logger.debug(f"Failed to load {annot_path}: {e}")

    # Use annotation if available
    if annotation_df is not None:
        # Map probe IDs to gene symbols
        gene_symbols: List[Optional[str]] = []
        for probe_id in probe_ids:
            if probe_id in annotation_df.index:
                symbol = annotation_df.loc[probe_id, 'Gene Symbol']
                if pd.notna(symbol) and symbol:
                    # Handle multiple symbols (take first)
                    symbol = str(symbol).split('///')[0].split(';')[0].strip()
                    gene_symbols.append(symbol)
                else:
                    gene_symbols.append(None)
            else:
                gene_symbols.append(None)

        feature_metadata = pd.DataFrame(
            {
                'probe_id': probe_ids,
                'gene_symbol': gene_symbols,
            },
            index=probe_ids
        )
        logger.info(f"Mapped {sum(pd.notna(gene_symbols))} probes to gene symbols using annotation file.")
    else:
        # Placeholder mapping
        feature_metadata = pd.DataFrame(
            {
                'probe_id': probe_ids,
                'gene_symbol': [pid.split('_')[0] if '_' in pid else pid
                               for pid in probe_ids],  # Placeholder
            },
            index=probe_ids
        )

        if platform_id:
            logger.warning(
                f"Using placeholder probe-to-gene mapping for {platform_id}. "
                f"To get proper mapping, download annotation file from GEO: "
                f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={platform_id} "
                f"and save as '{platform_id}.annot' or '{platform_id}.txt' in your datasets directory."
            )
        else:
            logger.warning(
                "Using placeholder probe-to-gene mapping. "
                "For production, load actual platform annotation files."
            )

    return feature_metadata


def _load_geo_annotation_file(annotation_file: Path) -> pd.DataFrame:
    """
    Load a GEO platform annotation file.

    Handles common formats:
    - Tab-separated with ID and Gene Symbol columns
    - May have comment lines starting with # before header
    - May have header or not

    Parameters
    ----------
    annotation_file : Path
        Path to annotation file

    Returns
    -------
    pd.DataFrame
        Annotation data with probe IDs as index and 'Gene Symbol' column
    """
    # Try different separators and encodings
    for sep in ['\t', ',', ';']:
        try:
            # Try reading with comment lines skipped
            df = pd.read_csv(
                annotation_file,
                sep=sep,
                low_memory=False,
                encoding='utf-8',
                comment='#',  # Skip lines starting with #
                on_bad_lines='skip'  # Skip malformed lines
            )

            # If we got an empty dataframe, try without comment skipping
            if df.empty:
                df = pd.read_csv(annotation_file, sep=sep, low_memory=False, encoding='utf-8')

            # Look for ID and Gene Symbol columns (case-insensitive)
            id_col = None
            symbol_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                # Match ID column (should be first column or contain "ID" or "probe")
                if id_col is None:
                    if col_lower in ['id', 'id_ref', 'probe_id', 'probeset_id'] or \
                       (col_lower.startswith('id') and 'probe' in col_lower):
                        id_col = col
                # Match Gene Symbol column
                if symbol_col is None:
                    if 'gene' in col_lower and 'symbol' in col_lower:
                        symbol_col = col

            if id_col and symbol_col:
                # Remove rows where ID is missing
                df = df[df[id_col].notna()]
                df = df.set_index(id_col)
                if symbol_col != 'Gene Symbol':
                    df = df.rename(columns={symbol_col: 'Gene Symbol'})
                return df[['Gene Symbol']]
        except Exception as e:
            logger.debug(f"Failed to parse with separator '{sep}': {e}")
            continue

    raise ValueError(f"Could not parse annotation file {annotation_file}. Expected tab-separated file with ID and Gene Symbol columns.")


def load_proteomics_csv(
    filepath: Union[str, Path],
    gene_symbol_col: str = "Description",
    sample_cols_start: int = 8,  # Columns after metadata
    extract_gene_from_description: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load proteomics data from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to proteomics CSV file
    gene_symbol_col : str, default="Description"
        Column containing gene information (e.g., "GN=GENE_NAME" in description)
    sample_cols_start : int, default=8
        Index of first sample column (after metadata columns)
    extract_gene_from_description : bool, default=True
        If True, extracts gene symbol from description field (e.g., "GN=GENE_NAME")

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (expression_matrix, sample_metadata)
        - expression_matrix: proteins × samples (index: gene symbols, columns: sample IDs)
        - sample_metadata: sample-level metadata
    """
    filepath = Path(filepath)
    logger.info(f"Loading proteomics data from: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath, low_memory=False)

    # Extract gene symbols from description
    if extract_gene_from_description and gene_symbol_col in df.columns:
        gene_symbols: List[Optional[str]] = []
        for desc in df[gene_symbol_col]:
            if pd.notna(desc):
                # Extract "GN=GENE_NAME" pattern
                match = re.search(r'GN=(\w+)', str(desc))
                if match:
                    gene_symbols.append(match.group(1))
                else:
                    gene_symbols.append(None)
            else:
                gene_symbols.append(None)
        df['gene_symbol'] = gene_symbols
    else:
        # Use Accession or Protein ID as fallback
        if 'Accession' in df.columns:
            df['gene_symbol'] = df['Accession'].str.split('|').str[0]
        else:
            df['gene_symbol'] = df.index.astype(str)

    # Filter out rows without gene symbols
    df = df[df['gene_symbol'].notna()].copy()

    # Identify sample columns (numeric columns after metadata)
    all_cols = df.columns.tolist()
    sample_cols: List[str] = []
    for i, col in enumerate(all_cols):
        if i >= sample_cols_start:
            # Check if column contains numeric data
            try:
                pd.to_numeric(df[col], errors='raise')
                sample_cols.append(col)
            except (ValueError, TypeError):
                pass

    if not sample_cols:
        raise ValueError("Could not identify sample columns. Check sample_cols_start parameter.")

    # Build expression matrix
    expression_df = df.set_index('gene_symbol')[sample_cols].copy()

    # Aggregate multiple proteins per gene (take mean)
    expression_df = expression_df.groupby(expression_df.index).mean()

    # Build sample metadata
    # Infer condition from sample names (e.g., "A1.1" = Control, "PD1.1" = PD)
    conditions: List[str] = []
    for col in sample_cols:
        if col.startswith('A') or 'Control' in col or 'CTRL' in col.upper():
            conditions.append('CONTROL')
        elif 'PD' in col.upper() or 'DISEASE' in col.upper():
            conditions.append('PD')
        else:
            conditions.append('UNKNOWN')

    sample_metadata = pd.DataFrame(
        {'condition': conditions},
        index=sample_cols
    )

    logger.info(
        f"Loaded proteomics data: {expression_df.shape[0]} genes, "
        f"{expression_df.shape[1]} samples"
    )

    return expression_df, sample_metadata


def load_rna_brain_data(
    data_dir: Union[str, Path],
    dataset_files: Optional[List[Union[str, Path]]] = None,
) -> OmicsData:
    """
    Load RNA brain tissue data from GEO Series Matrix files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing RNA brain datasets
    dataset_files : List[str], optional
        List of specific files to load. If None, loads all .txt files in directory.

    Returns
    -------
    OmicsData
        RNA omics data container
    """
    data_dir = Path(data_dir)

    if dataset_files is None:
        # Find all series matrix files
        dataset_files = [str(p) for p in data_dir.glob("*_series_matrix.txt")]
        if not dataset_files:
            raise FileNotFoundError(f"No series matrix files found in {data_dir}")

    logger.info(f"Loading RNA brain data from {len(dataset_files)} files...")

    # Load and combine datasets
    all_expression: List[pd.DataFrame] = []
    all_sample_metadata: List[pd.DataFrame] = []
    all_feature_metadata: List[pd.DataFrame] = []

    platform_ids: List[str] = []
    for filepath in dataset_files:
        # Convert to Path if string
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        expr, sample_meta, platform_id = parse_geo_series_matrix(filepath, extract_platform_id=True)

        # Map probe IDs to gene symbols
        # Look for annotation files in the data directory or parent datasets directory
        annotation_dir = data_dir  # Try same directory first
        if not any(annotation_dir.glob(f"{platform_id}*")) and data_dir.parent.exists():
            annotation_dir = data_dir.parent  # Try parent directory (datasets/)
        feature_meta = map_probe_to_gene_symbol(
            expr.index.tolist(),
            platform_id=platform_id,
            annotation_dir=annotation_dir
        )

        all_expression.append(expr)
        all_sample_metadata.append(sample_meta)
        all_feature_metadata.append(feature_meta)
        if platform_id:
            platform_ids.append(platform_id)

    # Combine datasets (assuming same probe IDs across datasets)
    # For different platforms, would need more sophisticated merging
    if len(all_expression) > 1:
        # Find common probes
        common_probes = set(all_expression[0].index)
        for expr in all_expression[1:]:
            common_probes &= set(expr.index)

        logger.info(f"Found {len(common_probes)} common probes across datasets")

        # Filter to common probes and concatenate samples
        combined_expr = pd.concat(
            [expr.loc[list(common_probes)] for expr in all_expression],
            axis=1
        )
        combined_sample_meta = pd.concat(all_sample_metadata, axis=0)
        combined_feature_meta = all_feature_metadata[0].loc[list(common_probes)]
    else:
        combined_expr = all_expression[0]
        combined_sample_meta = all_sample_metadata[0]
        combined_feature_meta = all_feature_metadata[0]

    return OmicsData(
        expression=combined_expr,
        sample_metadata=combined_sample_meta,
        feature_metadata=combined_feature_meta,
        omic_type="rna"
    )


def load_proteomics_brain_data(
    data_dir: Union[str, Path],
    dataset_files: Optional[List[Union[str, Path]]] = None,
) -> OmicsData:
    """
    Load proteomics brain tissue data from CSV files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing proteomics datasets
    dataset_files : List[str], optional
        List of specific files to load. If None, loads RR_proteins-C1-QUANT.csv

    Returns
    -------
    OmicsData
        Proteomics omics data container
    """
    data_dir = Path(data_dir)

    if dataset_files is None:
        # Default to main quant file
        default_file = data_dir / "RR_proteins-C1-QUANT.csv"
        if not default_file.exists():
            raise FileNotFoundError(f"Proteomics file not found: {default_file}")
        dataset_files = [str(default_file)]

    logger.info(f"Loading proteomics brain data from {len(dataset_files)} files...")

    # Load and combine datasets
    all_expression: List[pd.DataFrame] = []
    all_sample_metadata: List[pd.DataFrame] = []

    for filepath in dataset_files:
        # Convert to Path if string
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        expr, sample_meta = load_proteomics_csv(filepath)
        all_expression.append(expr)
        all_sample_metadata.append(sample_meta)

    # Combine datasets
    if len(all_expression) > 1:
        # Find common genes
        common_genes = set(all_expression[0].index)
        for expr in all_expression[1:]:
            common_genes &= set(expr.index)

        logger.info(f"Found {len(common_genes)} common genes across datasets")

        combined_expr = pd.concat(
            [expr.loc[list(common_genes)] for expr in all_expression],
            axis=1
        )
        combined_sample_meta = pd.concat(all_sample_metadata, axis=0)
    else:
        combined_expr = all_expression[0]
        combined_sample_meta = all_sample_metadata[0]

    # Create feature metadata
    feature_metadata = pd.DataFrame(
        {'gene_symbol': combined_expr.index},
        index=combined_expr.index
    )

    return OmicsData(
        expression=combined_expr,
        sample_metadata=combined_sample_meta,
        feature_metadata=feature_metadata,
        omic_type="proteomics"
    )


def integrate_multiomics(
    rna_data: Optional[OmicsData] = None,
    proteomics_data: Optional[OmicsData] = None,
    meth_data: Optional[OmicsData] = None,
    min_omics_per_gene: int = 2,
) -> MultiOmicsData:
    """
    Integrate multiple omics datasets at the gene level.

    Finds common genes across omics and creates a unified structure.

    Parameters
    ----------
    rna_data : OmicsData, optional
        RNA expression data
    proteomics_data : OmicsData, optional
        Proteomics data
    meth_data : OmicsData, optional
        Methylation data (for future use)
    min_omics_per_gene : int, default=2
        Minimum number of omics a gene must appear in to be included

    Returns
    -------
    MultiOmicsData
        Integrated multi-omics data container
    """
    logger.info("Integrating multi-omics data at gene level...")

    # Collect gene sets from each omic
    gene_sets: Dict[str, set[str]] = {}
    if rna_data is not None:
        # Extract gene symbols from RNA data
        if 'gene_symbol' in rna_data.feature_metadata.columns:
            rna_genes = set(rna_data.feature_metadata['gene_symbol'].dropna())
        else:
            # Fallback: use index if no gene symbols
            rna_genes = set(rna_data.expression.index)
        gene_sets['rna'] = rna_genes
        logger.info(f"RNA: {len(rna_genes)} genes")

    if proteomics_data is not None:
        # Proteomics already has gene symbols as index
        prot_genes = set(proteomics_data.expression.index)
        gene_sets['proteomics'] = prot_genes
        logger.info(f"Proteomics: {len(prot_genes)} genes")

    if meth_data is not None:
        # Similar to RNA
        if 'gene_symbol' in meth_data.feature_metadata.columns:
            meth_genes = set(meth_data.feature_metadata['gene_symbol'].dropna())
        else:
            meth_genes = set(meth_data.expression.index)
        gene_sets['meth'] = meth_genes
        logger.info(f"Meth: {len(meth_genes)} genes")

    # Find common genes
    if not gene_sets:
        raise ValueError("No omics data provided for integration.")

    common_genes = set.intersection(*gene_sets.values())

    # Also include genes that appear in at least min_omics_per_gene omics
    all_genes = set.union(*gene_sets.values())
    gene_counts = {gene: sum(gene in gs for gs in gene_sets.values())
                   for gene in all_genes}
    filtered_genes = {g for g, count in gene_counts.items()
                     if count >= min_omics_per_gene}

    # If no common genes found, try case-insensitive matching
    if not common_genes and len(gene_sets) > 1:
        logger.warning("No exact gene symbol matches found. Attempting case-insensitive matching...")
        # Convert all to uppercase for matching
        rna_upper = {g.upper() for g in gene_sets.get('rna', set())}
        prot_upper = {g.upper() for g in gene_sets.get('proteomics', set())}
        common_upper = rna_upper & prot_upper

        if common_upper:
            # Map back to original case (prefer proteomics case as it's likely more standard)
            prot_genes_dict = {g.upper(): g for g in gene_sets.get('proteomics', set())}
            common_genes = {prot_genes_dict.get(g, g) for g in common_upper}
            logger.info(f"Found {len(common_genes)} genes using case-insensitive matching.")

    final_genes = common_genes if common_genes else filtered_genes

    if not final_genes:
        logger.warning(
            f"No common genes found across omics. This may be due to: "
            f"1) Missing probe-to-gene annotation for RNA data, "
            f"2) Different gene identifier systems, or "
            f"3) No actual overlap between datasets. "
            f"Consider using actual platform annotation files for RNA probe mapping."
        )
        # Fallback: use all genes from each omic (will create separate graphs)
        final_genes = all_genes
        logger.info(f"Using all genes from all omics ({len(final_genes)} total) for separate graph construction.")
    else:
        logger.info(
            f"Found {len(final_genes)} genes common across omics "
            f"(min_omics={min_omics_per_gene})"
        )

    return MultiOmicsData(
        rna=rna_data,
        proteomics=proteomics_data,
        meth=meth_data,
        common_genes=list(final_genes)
    )


def load_multiomics_brain_data(
    base_dir: Union[str, Path],
    rna_dir: Optional[Union[str, Path]] = None,
    proteomics_dir: Optional[Union[str, Path]] = None,
    meth_dir: Optional[Union[str, Path]] = None,
) -> MultiOmicsData:
    """
    Convenience function to load all brain tissue multi-omics data.

    Parameters
    ----------
    base_dir : str or Path
        Base directory containing omics subdirectories
    rna_dir : str or Path, optional
        Directory for RNA data (default: base_dir/rna/brain)
    proteomics_dir : str or Path, optional
        Directory for proteomics data (default: base_dir/proteomics/brain)
    meth_dir : str or Path, optional
        Directory for methylation data (default: base_dir/meth/brain)

    Returns
    -------
    MultiOmicsData
        Integrated multi-omics data
    """
    base_dir = Path(base_dir)

    # Set default directories
    if rna_dir is None:
        rna_dir = base_dir / "rna" / "brain"
    if proteomics_dir is None:
        proteomics_dir = base_dir / "proteomics" / "brain"
    if meth_dir is None:
        meth_dir = base_dir / "meth" / "brain"

    # Load each omic
    rna_data = None
    proteomics_data = None
    meth_data = None

    # Convert to Path for .exists() check
    rna_dir_path = Path(rna_dir) if isinstance(rna_dir, str) else rna_dir
    proteomics_dir_path = Path(proteomics_dir) if isinstance(proteomics_dir, str) else proteomics_dir
    meth_dir_path = Path(meth_dir) if isinstance(meth_dir, str) else meth_dir

    if rna_dir_path.exists():
        try:
            rna_data = load_rna_brain_data(rna_dir_path)
        except Exception as e:
            logger.warning(f"Failed to load RNA data: {e}")

    if proteomics_dir_path.exists():
        try:
            proteomics_data = load_proteomics_brain_data(proteomics_dir_path)
        except Exception as e:
            logger.warning(f"Failed to load proteomics data: {e}")

    if meth_dir_path.exists():
        try:
            # Placeholder for methylation (not implemented yet)
            logger.info("Methylation loading not yet implemented")
        except Exception as e:
            logger.warning(f"Failed to load methylation data: {e}")

    # Integrate
    multiomics = integrate_multiomics(
        rna_data=rna_data,
        proteomics_data=proteomics_data,
        meth_data=meth_data,
    )

    return multiomics
