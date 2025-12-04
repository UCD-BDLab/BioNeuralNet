# GraphTransformer for BioNeuralNet

This document provides a focused overview and usage guide for the GraphTransformer model integrated into BioNeuralNet, along with a map of the Python files changed as part of this effort.

## Overview

GraphTransformer is a transformer-style architecture for graph-structured data that leverages attention over node features and optional positional encodings to capture structural information.

Key highlights:
- Transformer-style multi-head attention for graphs
- Flexible positional encoding (currently learnable; Laplacian eigenvectors planned)
- Drop-in usage via `GNNEmbedding(model_type='GRAPHTRANSFORMER')`

## Setup

- Install base dependencies:

```bash
pip install -r requirements.txt
```

- Install PyTorch and PyTorch Geometric (separately):

```bash
pip install torch
pip install torch_geometric
```

- Optional: dev and docs tooling:

```bash
pip install -r requirements-dev.txt
pip install -r docs/requirements.txt
```

## Positional Encoding

Current behavior:
- Learnable positional vectors are added to node embeddings inside `PositionalEncoding`.

Planned enhancement:
- Laplacian eigenvector positional encoding (compute per-graph normalized Laplacian eigenvectors; project via the existing linear layer; exclude trivial components; handle batching per graph).

## Quick Start with GNNEmbedding

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bioneuralnet.utils.graph import gen_gaussian_knn_graph
from bioneuralnet.network_embedding.gnn_embedding import GNNEmbedding

# Prepare synthetic omics and target
n_samples, n_features = 100, 50
omics = pd.DataFrame(np.random.normal(0, 1, size=(n_samples, n_features)),
                     columns=[f"f_{i}" for i in range(n_features)])
omics = pd.DataFrame(StandardScaler().fit_transform(omics),
                     columns=omics.columns, index=omics.index)
w = np.random.normal(0, 1, size=n_features)
target = pd.DataFrame({"phenotype": omics.values.dot(w) + np.random.normal(0, 0.5, size=n_samples)},
                      index=omics.index)

# Build adjacency (Gaussian kNN on features)
adjacency = gen_gaussian_knn_graph(omics, k=20)
X_train, X_test, y_train, y_test = train_test_split(omics, target, test_size=0.2, random_state=42)

# Initialize GraphTransformer via GNNEmbedding
embedder = GNNEmbedding(
    adjacency_matrix=adjacency,
    omics_data=X_train,
    phenotype_data=y_train,
    model_type='GRAPHTRANSFORMER',
    hidden_dim=128,
    layer_num=3,
    num_epochs=100,
    lr=1e-3,
    dropout=0.1,
    activation='gelu',
)

embedder.fit()
E = embedder.embed(as_df=False)  # node embeddings (torch.Tensor)
```

## Direct Usage with PyTorch Geometric

```python
import torch
from torch_geometric.data import Data
from bioneuralnet.network_embedding.graph_transformer import GraphTransformer

# Create a small synthetic graph
e = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # edge_index
x = torch.randn(3, 128)  # node features

# Build the PyG Data object
data = Data(x=x, edge_index=e)

# Initialize GraphTransformer
model = GraphTransformer(
    input_dim=128,
    hidden_dim=128,
    layer_num=3,
    heads=4,
    dropout=0.1,
    final_layer="regression",
    activation="gelu",
)

model.train()
output = model(data)  # forward pass; returns per-node outputs
embeddings = model.get_embeddings(data)  # node embeddings before final head
```

## Evaluation Script

You can run the example evaluation script to compare GraphTransformer behavior on synthetic or prepared data:
- Module: `examples.graph_transformer_evaluation`

Usage example:

```bash
python -m examples.graph_transformer_evaluation

# Optional attention heatmap flags
python -m examples.graph_transformer_evaluation --full-heatmap
python -m examples.graph_transformer_evaluation --full-heatmap --full-heatmap-max-n 2048
python -m examples.graph_transformer_evaluation --full-heatmap-force --full-heatmap-dpi 300
```

## Dataset Preparation

- Download TCGA and STRING assets (defaults to BRCA and downloads both when no flags are given):

```bash
python -m examples.download_tcga_data --cancer-type BRCA --tcga --string --output-dir bioneuralnet/datasets/brca
```

- Preprocess TCGA + STRING into harmonized inputs for examples and tutorials:

```bash
python -m examples.preprocess_tcga_string
```

## Classification Metrics

If phenotype labels are categorical (e.g., PAM50 subtypes), the evaluation script computes Macro F1 and Macro AUC using a logistic regression classifier on the learned sample embeddings. Result plots are saved to `visualization_results`:

- `graph_transformer_classification_f1.png`
- `graph_transformer_classification_auc.png`

Regression metrics (R², RMSE) remain available and are saved as:

- `graph_transformer_comparison.png`

## Attention Visualization

For interpretability, the GraphTransformer caches the last layer’s attention weights during a forward pass. You can access them via:

```python
atts = model.get_last_attentions()  # list per layer; each dict includes 'alpha'
```

When running `examples.graph_transformer_evaluation`, an attention heatmap (averaged across heads) is generated and saved as:

- `graph_transformer_attention_heatmap.png`

Additional visuals produced by the evaluation script:

- `graph_transformer_attention_heatmap_binned.png`
- `graph_transformer_attention_edges_scatter.png`
- `graph_transformer_attention_heatmap_topk.png`

This helps inspect which edges (gene–gene interactions) receive higher attention in the model.

## Tests

Unit tests include forward correctness, shape checks, numeric stability, and gradients:
- `tests/test_graph_transformer.py`

Run all tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## API Documentation

Full API is documented via Sphinx (napoleon + autodoc + autosummary). After building docs, refer to:
- Module: `bioneuralnet.network_embedding.graph_transformer`
- Classes: `GraphTransformer`, `GraphTransformerLayer`, `PositionalEncoding`


## Troubleshooting

- If you see NaNs, try lowering learning rate and/or increasing dropout.
- For large graphs, consider precomputing Laplacian eigenvectors offline to avoid runtime overhead.
- Ensure `torch` and `torch_geometric` versions match those in `requirements.txt` and your CUDA setup.

## Suggested Defaults

- hidden_dim: 128
- layer_num: 3
- dropout: 0.1
- heads (GraphTransformer direct usage): 4

## Additional Modified or Referenced Python Files

- bioneuralnet/network_embedding/gnn_embedding.py
  - Wiring for `model_type='GRAPHTRANSFORMER'` and integration with the embedding pipeline, including shape handling and model initialization nuances.

- bioneuralnet/network_embedding/gnn_models.py
  - Ensured GraphTransformer is imported/registered so the factory can construct it via `model_type`.

- examples/graph_transformer_evaluation.py
  - Script to run/evaluate GraphTransformer on synthetic or prepared datasets.
  - Supports attention visualization flags: `--full-heatmap`, `--full-heatmap-max-n`, `--full-heatmap-force`, `--full-heatmap-dpi`; saves results under `visualization_results/` (e.g., `graph_transformer_comparison.png`).

- examples/download_tcga_data.py
  - Utility to fetch TCGA clinical/gene expression and STRING PPI resources used by examples/tests.

- examples/preprocess_tcga_string.py
  - Preprocessing pipeline to merge TCGA and STRING resources; prepares artifacts for evaluation and tutorials.

- examples/preprocessing_example.py
  - Example-level checks and visualizations validating the preprocessing pipeline outputs.

- examples/test_synthetic_data.py
  - Synthetic data generation and quick validation used for model sanity checks and tutorial demos.
