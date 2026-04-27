"""BioNeuralNet: Graph Neural Network-based Multi-Omics Network Data Analysis.

BioNeuralNet is a modular framework tailored for end-to-end network-based multi-omics data analysis. It leverages Graph Neural Networks (GNNs) to transform complex molecular networks into biologically meaningful low-dimensional representations, enabling diverse downstream analytical tasks.

Key Features:

* **Network Construction**: Modules to construct networks from raw tabular data using similarity, correlation, neighborhood-based, or phenotype-driven strategies (e.g., SmCCNet).
* **Network Embedding**: Generate low-dimensional representations using advanced Graph Neural Networks, including GCN, GAT, GraphSAGE, and GIN.
* **Subgraph Detection**: Identify biologically meaningful modules using supervised and unsupervised community detection methods like Correlated Louvain and PageRank.
* **Downstream Tasks**: Execute specialized pipelines such as DPMON (Disease Prediction using Multi-Omics Networks) and Subject Representation for patient-level analysis.
* **Data Handling**: Streamline data ingestion, feature selection (ANOVA, Random Forest), and preprocessing.
* **Reproducibility**: Built-in logging, configuration, and seeding utilities to ensure reproducible research.

"""

__version__ = "1.3.1"

# submodules to enable direct imports such as `from bioneuralnet import utils`
from . import utils
from . import metrics
from . import network
from . import datasets
from . import clustering
from . import network_embedding
from . import downstream_task
from . import external_tools

from .network_embedding import GNNEmbedding
from .network import auto_pysmccnet
from .downstream_task import SubjectRepresentation, DPMON
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
    load_kipan
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
    "network",
    "external_tools",

    "GNNEmbedding",
    "SubjectRepresentation",
    "auto_pysmccnet",
    "DPMON",

    "DatasetLoader",
    "CorrelatedPageRank",
    "CorrelatedLouvain",

    "HybridLouvain",

    "load_example",
    "load_monet",
    "load_brca",
    "load_lgg",
    "load_kipan",

    "set_seed",
    "get_logger",
]
