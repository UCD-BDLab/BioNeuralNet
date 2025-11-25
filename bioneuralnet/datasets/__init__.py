"""Built-in datasets for BioNeuralNet.

This module provides specific loader functions for each available multi-omics dataset.
"""

from .dataset_loader import DatasetLoader

def load_example() -> dict:
    """Load the synthetic Example dataset.

    Returns:

        dict: Keys include 'X1', 'X2', 'Y', 'clinical'.
    """
    return DatasetLoader("example").data

def load_monet() -> dict:
    """Load the synthetic MONET dataset.

    Returns:

        dict: Keys include 'gene', 'mirna', 'phenotype', 'rppa', 'clinical'.
    """
    return DatasetLoader("monet").data

def load_brca() -> dict:
    """Load the Breast Invasive Carcinoma (BRCA) dataset.

    Returns:

        dict: Keys include 'mirna', 'target', 'clinical', 'rna', 'meth'.
    """
    return DatasetLoader("brca").data

def load_lgg() -> dict:
    """Load the Brain Lower Grade Glioma (LGG) dataset.

    Returns:

        dict: Keys include 'mirna', 'target', 'clinical', 'rna', 'meth'.
    """
    return DatasetLoader("lgg").data

def load_kipan() -> dict:
    """Load the Pan-kidney Cohort (KIPAN) dataset.

    Returns:

        dict: Keys include 'mirna', 'target', 'clinical', 'rna', 'meth'.
    """
    return DatasetLoader("kipan").data

def load_paad() -> dict:
    """Load the Pancreatic Adenocarcinoma (PAAD) dataset.

    Returns:

        dict: Keys include 'cnv', 'target', 'clinical', 'rna', 'meth'.
    """
    return DatasetLoader("paad").data

__all__ = [
    "DatasetLoader",
    "load_example",
    "load_monet",
    "load_brca",
    "load_lgg",
    "load_kipan",
    "load_paad",
]
