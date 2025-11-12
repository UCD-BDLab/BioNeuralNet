import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # ensuring headless plotting

from examples.graph_transformer_evaluation import (
    plot_embeddings,
    plot_performance,
    plot_classification_metrics,
)


def _mock_results():
    # Building a small synthetic result dict compatible with plotting helpers
    rng = np.random.RandomState(0)
    emb1 = rng.randn(20, 8)
    emb2 = rng.randn(15, 8)
    y1 = rng.rand(20)
    y2 = rng.rand(15)
    return {
        "RawRidge": {
            "embeddings": emb1,
            "y": y1,
            "valid": True,
            "r2": 0.42,
            "mse": 0.58,
        },
        "GraphTransformer": {
            "embeddings": emb2,
            "y": y2,
            "valid": True,
            "r2": 0.55,
            "mse": 0.45,
            "f1": 0.71,
            "auc": 0.80,
        },
    }


def test_plot_embeddings(tmp_path):
    results = _mock_results()
    models = ["RawRidge", "GraphTransformer"]
    plot_embeddings(results, models, tmp_path)
    out = tmp_path / "embedding_visualizations.png"
    assert out.exists(), "embedding_visualizations.png should be created"
    assert os.path.getsize(out) > 0, "embedding_visualizations.png should be non-empty"


def test_plot_performance(tmp_path):
    results = _mock_results()
    models = ["RawRidge", "GraphTransformer"]
    plot_performance(results, models, tmp_path)
    out = tmp_path / "graph_transformer_comparison.png"
    assert out.exists(), "graph_transformer_comparison.png should be created"
    assert os.path.getsize(out) > 0, "graph_transformer_comparison.png should be non-empty"


def test_plot_classification_metrics(tmp_path):
    results = _mock_results()
    models = ["RawRidge", "GraphTransformer"]
    plot_classification_metrics(results, models, tmp_path)
    f1_out = tmp_path / "graph_transformer_classification_f1.png"
    auc_out = tmp_path / "graph_transformer_classification_auc.png"
    # F1 and AUC plots only require at least one model with metrics
    assert f1_out.exists(), "graph_transformer_classification_f1.png should be created"
    assert os.path.getsize(f1_out) > 0, "graph_transformer_classification_f1.png should be non-empty"
    assert auc_out.exists(), "graph_transformer_classification_auc.png should be created"
    assert os.path.getsize(auc_out) > 0, "graph_transformer_classification_auc.png should be non-empty"
