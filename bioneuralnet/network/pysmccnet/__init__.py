r"""
Sparse Multiple Canonical Correlation Network (SmCCNet 2.0).

This module implements the SmCCNet pipeline for multi-omics network inference
using PyTorch with optional CUDA acceleration. It integrates multiple omics
data types to construct sparse biological networks associated with a phenotype
of interest.

Developed in collaboration with the Kechris Lab at CU Anschutz.

References:
    Liu et al. (2024), "SmCCNet 2.0: A Comprehensive Tool for Multi-omics
    Network Inference with Shiny Visualization," BMC Bioinformatics.

    Shi et al. (2019), "Unsupervised Discovery of Phenotype-Relevant
    Multi-omics Networks," Bioinformatics.

    Vu et al. (2023), "NetSHy: Network Summarization via a Hybrid Approach
    Leveraging Topological Properties," Bioinformatics.

Notes:
    **Sparse Canonical Correlation Analysis (SCCA)**
    The core optimization finds sparse weight vectors that maximize
    cross-covariance between data matrices under L1/L2 constraints:

    .. math::
        \\max_{u,v} \\; u^T X^T Y v

        \\text{s.t.} \\quad \\|u\\|_2 \\leq 1, \\; \\|v\\|_2 \\leq 1,
        \\; \\|u\\|_1 \\leq c_1, \\; \\|v\\|_1 \\leq c_2

    Where :math:`X, Y` are standardized data matrices and :math:`c_1, c_2`
    control sparsity via L1 penalty parameters.

    **Multi-omics SmCCA Objective**
    Extends SCCA to jointly maximize pairwise correlations across multiple
    omics layers with scaling factors:

    .. math::
        \\max \\; \\sum_{i < j} w_{ij} \\cdot u_i^T X_i^T X_j u_j

    Where :math:`w_{ij}` balances the contribution of each omics pair and
    :math:`u_i` is the sparse weight vector for omics layer :math:`i`.

    **Sparse PLS-DA (Binary Phenotype)**
    For binary or categorical phenotypes, sparse partial least squares
    discriminant analysis extracts latent factors via soft-thresholded
    direction estimation:

    .. math::
        w_{\\text{new}} = S\\bigl(X^T y, \\; \\eta \\cdot \\max|X^T y|\\bigr)

        S(z, \\lambda) = \\operatorname{sign}(z) \\cdot \\max(|z| - \\lambda, \\; 0)

    Latent factors are then weighted by logistic regression coefficients
    to produce phenotype-discriminative canonical weights.

    **Network Adjacency Construction**
    The global similarity matrix is built by averaging outer products of
    absolute weight vectors across subsampling iterations:

    .. math::
        \\bar{A} = \\frac{1}{K} \\sum_{k=1}^{K} |w^{(k)}| \\cdot |w^{(k)}|^T

    Where :math:`w^{(k)}` is the weight vector from the :math:`k`-th
    subsample and :math:`\\bar{A}` is normalized to a maximum of 1.0.

    **Network Module Extraction**
    Hierarchical clustering (complete linkage) on :math:`1 - \\bar{A}` with
    a user-defined cut height partitions features into modules. Modules are
    pruned to a target size range by iteratively removing the lowest-degree
    node, then summarized via the NetSHy hybrid approach (Laplacian-weighted
    PCA).

Algorithm:
    The automated pipeline proceeds through five sequential phases:

    1. **Preprocessing**: Optional covariate regression, centering, and
       scaling of each omics layer.

    2. **Scaling Factor Determination**: Pairwise canonical correlations
       between omics layers are computed and shrunk by a user-defined
       factor to balance between-omics and omics-phenotype contributions.

    3. **Cross-Validation**: K-fold CV over a penalty grid selects the
       sparsity parameters that minimize the ratio of prediction error
       to test canonical correlation (CCA) or maximize classification
       accuracy (PLS).

    4. **Subsampling**: The selected penalties are applied across repeated
       feature subsamples to construct a stable global adjacency matrix.

    5. **Module Extraction**: Hierarchical clustering and degree-based
       pruning produce final subnetworks, each summarized by NetSHy
       scores and their phenotype correlations.
"""

from .pipeline import auto_pysmccnet

__all__ = ["auto_pysmccnet"]
