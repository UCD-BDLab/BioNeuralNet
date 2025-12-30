"""BioNeuralNet: Graph Neural Network-based Multi-Omics Network Data Analysis.

BioNeuralNet is a flexible and modular framework tailored for end-to-end network-based multi-omics data analysis. It leverages Graph Neural Networks (GNNs) to transform complex molecular networks into biologically meaningful low-dimensional representations, enabling diverse downstream analytical tasks.

Key Features:

* **Network Construction**: Modules to construct networks from raw tabular data using similarity, correlation, neighborhood-based, or phenotype-driven strategies (e.g., SmCCNet).
* **Network Embedding**: Generate low-dimensional representations using advanced Graph Neural Networks, including GCN, GAT, GraphSAGE, and GIN.
* **Subgraph Detection**: Identify biologically meaningful modules using supervised and unsupervised community detection methods like Correlated Louvain and PageRank.
* **Downstream Tasks**: Execute specialized pipelines such as DPMON (Disease Prediction using Multi-Omics Networks) and Subject Representation for patient-level analysis.
* **Data Handling**: Streamline data ingestion, feature selection (ANOVA, Random Forest), and preprocessing.
* **Reproducibility**: Built-in logging, configuration, and seeding utilities to ensure reproducible research.

Modules:

* `network_embedding`: Generates network embeddings via GNN architectures.
* `subject_representation`: Fuses network embeddings with omics data for subject profiling.
* `downstream_task`: Contains pipelines for disease prediction (DPMON) and representation learning.
* `clustering`: Implements correlated and hybrid clustering algorithms for module detection.
* `external_tools`: Wraps external packages (e.g., SmCCNet) for network inference.
* `metrics`: Provides tools for correlation analysis, performance evaluation, and plotting.
* `datasets`: Access to synthetic and real-world (TCGA) multi-omics datasets.
* `utils`: Utilities for logging, reproducibility, graph generation, and data processing.
"""

__version__ = "1.2.2"

# submodules to enable direct imports such as `from bioneuralnet import utils`
from . import utils
from . import metrics
from . import datasets
from . import clustering
from . import network_embedding
from . import downstream_task
from . import external_tools

from .network_embedding import GNNEmbedding
from .downstream_task import SubjectRepresentation, DPMON
from .external_tools import SmCCNet
from .datasets import DatasetLoader

from .clustering import (
    CorrelatedPageRank,
    CorrelatedLouvain,
    HybridLouvain,
)

from .datasets import (
    load_example,
    load_monet,
    load_brca,
    load_lgg,
    load_kipan,
    load_paad,
)

from .utils import (
    set_seed,
    get_logger,
)

__all__ = [
    "__version__",

    "utils",
    "metrics",
    "datasets",
    "clustering",
    "network_embedding",
    "downstream_task",
    "external_tools",

    "GNNEmbedding",
    "SubjectRepresentation",
    "DPMON",
    "SmCCNet",
    "DatasetLoader",

    "CorrelatedPageRank",
    "CorrelatedLouvain",
    "HybridLouvain",

    "load_example",
    "load_monet",
    "load_brca",
    "load_lgg",
    "load_kipan",
    "load_paad",

    "set_seed",
    "get_logger",
]
