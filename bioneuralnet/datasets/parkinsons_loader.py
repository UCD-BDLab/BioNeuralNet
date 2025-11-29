from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParkinsonsData:
    """
    Container for Parkinson's disease (PD) transcriptomics data (GSE165082).

    Attributes
    ----------
    expression:
        Gene expression matrix with genes as rows and samples as columns.
        Index: Ensembl gene IDs (e.g., 'ENSG00000223972').
        Columns: sample identifiers (e.g., '011_PD', '021_CC').

    sample_metadata:
        Sample-level metadata.
        Index: sample identifiers (matching `expression` columns).
        Columns:
            - 'condition': 'PD' or 'CC'
            - 'raw_name': original column name (if different), optional.

    gene_metadata:
        Gene-level metadata.
        Index: Ensembl gene IDs (matching `expression` index).
        Typical columns (if annotation is provided):
            - 'Symbol'
            - 'Description'
            - 'GeneType'
            - and any other columns from the annotation table.
    """

    expression: pd.DataFrame
    sample_metadata: pd.DataFrame
    gene_metadata: pd.DataFrame


class ParkinsonsLoader:
    """
    Loader for the GSE165082 Parkinson's disease transcriptomics dataset.

    This class:

    - Loads the PD expression matrix (genes × samples) from
      ``GSE165082_PD-CC.counts.txt``.
    - Parses sample names into PD vs control labels based on suffix:
        - ``*_PD`` → condition == 'PD'
        - ``*_CC`` → condition == 'CC'
    - Optionally maps Ensembl gene IDs to gene symbols / annotations using
      ``Human.GRCh38.p13.annot.tsv``.

    By default, the loader assumes the BioNeuralNet repository layout:

    - Project root is three levels above this file
      (``bioneuralnet/datasets/parkinsons_loader.py`` → root).
    - PD data are stored in ``<root>/PD-Notebooks/datasets/``.

    You can override all paths via the constructor.
    """

    DEFAULT_COUNTS_FILE = "GSE165082_PD-CC.counts.txt"
    DEFAULT_ANNOTATION_FILE = "Human.GRCh38.p13.annot.tsv"

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        counts_path: Optional[str | Path] = None,
        annotation_path: Optional[str | Path] = None,
        use_annotation: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        data_dir:
            Base directory containing the PD dataset files.
            If ``None``, uses ``<project_root>/PD-Notebooks/datasets/``.

        counts_path:
            Optional explicit path to ``GSE165082_PD-CC.counts.txt``.
            If provided, this overrides ``data_dir`` for the expression file.

        annotation_path:
            Optional explicit path to ``Human.GRCh38.p13.annot.tsv``.
            If provided, this overrides ``data_dir`` for the annotation file.

        use_annotation:
            Whether to attempt loading and aligning gene annotations. If the
            file is missing or malformed, an empty ``gene_metadata`` table
            will be returned with the correct index.
        """
        # Heuristic project root: .../BioNeuralNet/
        self._root_dir = Path(__file__).resolve().parents[2]

        if data_dir is None:
            self.data_dir = self._root_dir / "PD-Notebooks" / "datasets"
        else:
            self.data_dir = Path(data_dir)

        self.counts_path = (
            Path(counts_path)
            if counts_path is not None
            else self.data_dir / self.DEFAULT_COUNTS_FILE
        )

        self.annotation_path = (
            Path(annotation_path)
            if annotation_path is not None
            else self.data_dir / self.DEFAULT_ANNOTATION_FILE
        )

        self.use_annotation = use_annotation

        logger.info(
            "Initialized ParkinsonsLoader with counts='%s', annotation='%s'",
            self.counts_path,
            self.annotation_path if use_annotation else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> ParkinsonsData:
        """
        Load expression matrix, sample labels and gene metadata.

        Returns
        -------
        ParkinsonsData
            Dataclass with:

            - ``expression``: genes × samples counts.
            - ``sample_metadata``: PD/control labels and sample info.
            - ``gene_metadata``: gene annotations aligned to expression index.
        """
        expression = self._load_expression()
        sample_metadata = self._build_sample_metadata(expression.columns)
        gene_metadata = self._load_gene_metadata(expression.index)

        return ParkinsonsData(
            expression=expression,
            sample_metadata=sample_metadata,
            gene_metadata=gene_metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_expression(self) -> pd.DataFrame:
        """
        Load the GSE165082 PD-CC counts matrix.

        The file is expected to have:

        - A ``Geneid`` column with Ensembl gene IDs.
        - Remaining columns for samples (e.g., ``011_PD``, ``021_CC``).
        """
        if not self.counts_path.is_file():
            raise FileNotFoundError(
                f"Counts file not found at '{self.counts_path}'. "
                "Please verify the path or provide 'counts_path' explicitly."
            )

        logger.info("Loading PD counts matrix from '%s'.", self.counts_path)
        df = pd.read_csv(self.counts_path, sep="\t")

        if "Geneid" not in df.columns:
            raise ValueError(
                "Expected a 'Geneid' column in the counts file, "
                f"but found columns: {list(df.columns[:5])}..."
            )

        df = df.set_index("Geneid")
        df.columns = df.columns.astype(str)

        logger.info(
            "Loaded counts matrix with shape %s (genes × samples).",
            df.shape,
        )
        return df

    def _build_sample_metadata(self, sample_names) -> pd.DataFrame:
        """
        Build sample metadata from expression column names.

        Parameters
        ----------
        sample_names:
            Iterable of sample column names from the expression matrix.
        """
        records = []
        for name in sample_names:
            condition = self._infer_condition_from_sample_name(name)
            records.append(
                {
                    "sample_id": name,
                    "condition": condition,
                    "raw_name": name,
                }
            )

        sample_df = pd.DataFrame.from_records(records).set_index("sample_id")
        logger.info(
            "Sample metadata constructed. Condition counts: %s",
            sample_df["condition"].value_counts().to_dict(),
        )
        return sample_df

    def _infer_condition_from_sample_name(self, name: str) -> str:
        """
        Infer PD vs Control labels from the sample name.

        Rules
        -----
        - names ending with ``'_PD'`` (case-insensitive) → ``'PD'``
        - names ending with ``'_CC'`` (case-insensitive) → ``'CC'``
        - otherwise → ``'unknown'``
        """
        lower = name.lower()
        if lower.endswith("_pd"):
            return "PD"
        if lower.endswith("_cc"):
            return "CC"

        logger.warning(
            "Could not infer condition for sample '%s'. "
            "Assigned label 'unknown'.",
            name,
        )
        return "unknown"

    def _load_gene_metadata(self, gene_ids) -> pd.DataFrame:
        """
        Load and align gene-level metadata from the annotation file.

        Parameters
        ----------
        gene_ids:
            Index (iterable) of gene IDs from the expression matrix.
        """
        index = pd.Index(gene_ids, name="EnsemblGeneID")

        if not self.use_annotation:
            logger.info("Annotation loading disabled ('use_annotation=False').")
            return pd.DataFrame(index=index)

        if not self.annotation_path.is_file():
            logger.warning(
                "Annotation file not found at '%s'. Returning empty "
                "gene_metadata.",
                self.annotation_path,
            )
            return pd.DataFrame(index=index)

        logger.info("Loading gene annotations from '%s'.", self.annotation_path)
        annot = pd.read_csv(self.annotation_path, sep="\t")

        if "EnsemblGeneID" not in annot.columns:
            logger.warning(
                "Expected 'EnsemblGeneID' column in the annotation file, "
                "but found columns: %s. Returning empty gene_metadata.",
                list(annot.columns[:8]),
            )
            return pd.DataFrame(index=index)

        # Handle duplicate EnsemblGeneID values
        # (common in annotation files due to multiple transcripts/isoforms per gene)
        duplicates = annot["EnsemblGeneID"].duplicated()
        if duplicates.any():
            n_duplicates = duplicates.sum()
            logger.info(
                "Found %d duplicate EnsemblGeneID entries. "
                "Keeping first occurrence for each gene.",
                n_duplicates,
            )
            annot = annot.drop_duplicates(subset="EnsemblGeneID", keep="first")

        annot = annot.set_index("EnsemblGeneID")
        annot_aligned = annot.reindex(index)

        logger.info(
            "Gene metadata aligned: %d genes, %d columns.",
            annot_aligned.shape[0],
            annot_aligned.shape[1],
        )
        return annot_aligned


def load_parkinsons_data(
    data_dir: Optional[str | Path] = None,
    counts_path: Optional[str | Path] = None,
    annotation_path: Optional[str | Path] = None,
    use_annotation: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load the GSE165082 Parkinson's dataset.

    Returns
    -------
    (expression, sample_metadata, gene_metadata)
        - expression: genes × samples counts matrix.
        - sample_metadata: PD vs control labels.
        - gene_metadata: gene annotations (may be empty).
    """
    loader = ParkinsonsLoader(
        data_dir=data_dir,
        counts_path=counts_path,
        annotation_path=annotation_path,
        use_annotation=use_annotation,
    )
    data = loader.load()
    return data.expression, data.sample_metadata, data.gene_metadata
