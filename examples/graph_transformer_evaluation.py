"""Graph Transformer Evaluation Script.

This example runs an end-to-end evaluation of GraphTransformer alongside
traditional GNN models (GCN, GAT, SAGE, GIN) and a raw Ridge baseline.

It demonstrates:
- Dataset loading (BRCA if available, otherwise synthetic)
- Preprocessing and KNN graph construction
- Model training and regression/classification metrics
- Embedding visualization (t-SNE/PCA fallback)
- Attention visualization (full, binned, scatter, and top-K)

The script is organized into functions with docstrings so our documentation
site can render them properly. All functions use Google-style docstrings.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Union
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bioneuralnet.network_embedding.gnn_embedding import GNNEmbedding
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.utils.graph import gen_gaussian_knn_graph
from bioneuralnet.metrics import evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

np.random.seed(42)
torch.manual_seed(42)
# Setting output directory once for all plots
VIZ_DIR = Path(__file__).resolve().parents[1] / "visualization_results"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Parse CLI arguments (parse_known_args to ignore unrelated flags)
parser = argparse.ArgumentParser(description="Graph Transformer evaluation", add_help=False)
parser.add_argument("--full-heatmap", action="store_true", help="Enable saving full NxN attention heatmap")
parser.add_argument("--full-heatmap-max-n", type=int, default=4096, help="Max nodes to allow full heatmap (default: 4096)")
parser.add_argument("--full-heatmap-force", action="store_true", help="Force full NxN heatmap regardless of size")
parser.add_argument("--full-heatmap-dpi", type=int, default=300, help="DPI for full heatmap image (default: 300)")
args, _unknown = parser.parse_known_args()

def plot_embeddings(results: Dict[str, Dict[str, Union[float, np.ndarray]]], models: List[str], out_dir: Path) -> None:
    """Visualize embeddings via t-SNE or PCA fallback and save a summary grid.

    Args:
        results: Per-model outputs containing ``embeddings`` and ``y``.
        models: Model names in display order.
        out_dir: Directory to save ``embedding_visualizations.png``.
    """
    try:
        from sklearn.manifold import TSNE
        import seaborn as sns  # noqa: F401 (kept for styling/consistency)
        plt.figure(figsize=(15, 10))
        for i, model_type in enumerate(models):
            try:
                emb = results.get(model_type, {}).get('embeddings', None)
                if emb is None:
                    plt.subplot(2, 3, i+1)
                    plt.text(0.5, 0.5, f"No embeddings for {model_type}", ha='center', va='center')
                    plt.axis('off')
                    continue
                emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
                n_samples = emb.shape[0]
                max_points = 1000
                y_arr = np.asarray(results.get(model_type, {}).get('y', []), dtype=np.float64)
                if n_samples > max_points:
                    idx = np.random.RandomState(42).choice(n_samples, size=max_points, replace=False)
                    emb_plot = emb[idx]
                    y_plot = y_arr[idx] if y_arr.size >= idx.size else y_arr
                else:
                    emb_plot = emb
                    y_plot = y_arr[:emb.shape[0]] if y_arr.size >= emb.shape[0] else y_arr
                col_std = emb_plot.std(axis=0)
                row_std = emb_plot.std(axis=1)
                degenerate = (np.all(col_std < 1e-12)) or (np.all(row_std < 1e-12)) or (float(np.var(emb_plot)) < 1e-12)
                plt.subplot(2, 3, i+1)
                if degenerate or emb_plot.shape[0] < 10:
                    if emb_plot.shape[0] >= 2:
                        n_components_pca = max(1, min(2, emb_plot.shape[1], emb_plot.shape[0]))
                        pca = PCA(n_components=n_components_pca, random_state=42)
                        try:
                            pca_2d = pca.fit_transform(emb_plot)
                            scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1] if pca_2d.shape[1] > 1 else np.zeros_like(pca_2d[:, 0]), c=y_plot, cmap='viridis')
                            plt.colorbar(scatter, label='Phenotype Value')
                            plt.title(f'{model_type} Sample Embeddings (PCA fallback)')
                            plt.xlabel('PC 1')
                            plt.ylabel('PC 2')
                        except Exception:
                            plt.text(0.5, 0.5, f"Degenerate embeddings for {model_type}", ha='center', va='center')
                            plt.axis('off')
                    else:
                        plt.text(0.5, 0.5, f"Insufficient data for {model_type}", ha='center', va='center')
                        plt.axis('off')
                else:
                    safe_perp = max(5, min(30, (emb_plot.shape[0] - 1) // 3))
                    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto', perplexity=safe_perp)
                    embeddings_2d = tsne.fit_transform(emb_plot)
                    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_plot, cmap='viridis')
                    plt.colorbar(scatter, label='Phenotype Value')
                    plt.title(f'{model_type} Sample Embeddings')
                    plt.xlabel('t-SNE 1')
                    plt.ylabel('t-SNE 2')
            except Exception as viz_e:
                print(f"Visualization failed for {model_type}: {viz_e}")
                continue
        plt.tight_layout()
        plt.savefig(out_dir / 'embedding_visualizations.png', dpi=300)
        print(f"Embedding visualizations saved to '{out_dir / 'embedding_visualizations.png'}'")
    except Exception as e:
        print(f"Could not generate embedding visualizations: {e}")

def plot_performance(results: Dict[str, Dict[str, Union[float, np.ndarray]]], models: List[str], out_dir: Path) -> None:
    """Plot R² comparison across valid models and save as an image.

    Args:
        results: Per-model outputs including regression scores.
        models: Model names in display order.
        out_dir: Directory to save 'graph_transformer_comparison.png'.
    """
    try:
        valid_models = [m for m in models if results.get(m, {}).get('valid', False)]
        plt.figure(figsize=(10, 6))
        if len(valid_models) > 0:
            r2s = [results[m]['r2'] for m in valid_models]
            mses = [results[m]['mse'] for m in valid_models]
            x = np.arange(len(valid_models))
            bars = plt.bar(x, r2s, color='steelblue', edgecolor='black')
            plt.xticks(x, valid_models, rotation=30, ha='right')
            plt.ylabel('R² score')
            plt.title('Graph Transformer Comparison (R²)')
            for i, b in enumerate(bars):
                plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{r2s[i]:.3f}", ha='center', va='bottom', fontsize=9)
            if (max(mses) - min(mses) < 1e-3) and (max(r2s) - min(r2s) < 1e-3):
                print("Warning: Metrics are nearly identical across models. This can indicate degenerate embeddings or a non-informative target.")
        else:
            if 'RawRidge' in results:
                plt.bar([0], [results['RawRidge']['r2']], color='steelblue', edgecolor='black')
                plt.xticks([0], ['RawRidge'])
                plt.ylabel('R² score')
                plt.title('Graph Transformer Comparison (R²)')
            else:
                plt.text(0.5, 0.5, "No valid model results to plot", ha='center', va='center')
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / 'graph_transformer_comparison.png', dpi=300)
        print(f"Comparison plot saved to '{out_dir / 'graph_transformer_comparison.png'}'")
    except Exception as plot_e:
        print(f"Could not generate graph_transformer_comparison plot: {plot_e}")

def plot_classification_metrics(results: Dict[str, Dict[str, Union[float, np.ndarray]]], models: List[str], out_dir: Path) -> None:
    """Plot Macro F1 and Macro AUC for classifiers when available.

    Args:
        results: Per-model outputs possibly including 'f1' and 'auc'.
        models: Model names in display order.
        out_dir: Directory to save classification figures.
    """
    try:
        valid_models = [m for m in models if results.get(m, {}).get('valid', False)]
        if len(valid_models) > 0 and any('f1' in results[m] for m in valid_models):
            plt.figure(figsize=(10, 6))
            f1s = [results[m].get('f1', np.nan) for m in valid_models]
            x = np.arange(len(valid_models))
            plt.bar(x, f1s, color='darkorange', edgecolor='black')
            plt.xticks(x, valid_models, rotation=30, ha='right')
            plt.ylabel('Macro F1')
            plt.title('Graph Transformer Comparison (Macro F1)')
            plt.tight_layout()
            plt.savefig(out_dir / 'graph_transformer_classification_f1.png', dpi=300)
            print(f"Classification F1 plot saved to '{out_dir / 'graph_transformer_classification_f1.png'}'")
            aucs = [results[m].get('auc', np.nan) for m in valid_models]
            plt.figure(figsize=(10, 6))
            plt.bar(x, aucs, color='seagreen', edgecolor='black')
            plt.xticks(x, valid_models, rotation=30, ha='right')
            plt.ylabel('Macro AUC')
            plt.title('Graph Transformer Comparison (Macro AUC)')
            plt.tight_layout()
            plt.savefig(out_dir / 'graph_transformer_classification_auc.png', dpi=300)
            print(f"Classification AUC plot saved to '{out_dir / 'graph_transformer_classification_auc.png'}'")
    except Exception as cls_plot_e:
        print(f"Could not generate classification plots: {cls_plot_e}")

def run_evaluation_cli():
    try:
        dataset = DatasetLoader("brca")
        omics_data = dataset.data["rna"]
        phenotype_data = dataset.data["pam50"]
        print("Loaded BRCA dataset")
    except Exception as e:
        print(f"Error loading BRCA dataset: {e}")
        print("Using synthetic data for demonstration")

        n_samples = 100
        n_features = 50

        omics_data = pd.DataFrame(
            np.random.normal(0, 1, size=(n_samples, n_features)),
            columns=[f"feature_{i}" for i in range(n_features)]
        )

        true_weights = np.random.normal(0, 1, size=n_features)
        phenotype_data = pd.DataFrame({
            'survival_time': np.dot(omics_data, true_weights) + np.random.normal(0, 0.5, size=n_samples)
        })

    omics_data = omics_data.apply(pd.to_numeric, errors='coerce')
    omics_data = omics_data.fillna(omics_data.mean())

    scaler_all = StandardScaler()
    omics_data = pd.DataFrame(
        scaler_all.fit_transform(omics_data),
        columns=omics_data.columns,
        index=omics_data.index,
    )
    if isinstance(phenotype_data, pd.DataFrame):
        target = pd.to_numeric(phenotype_data.iloc[:, 0], errors='coerce')
    else:
        target = pd.to_numeric(phenotype_data.squeeze(), errors='coerce')
    target = target.fillna(target.median())

    adjacency_matrix = gen_gaussian_knn_graph(omics_data, k=20)

    X_train, X_test, y_train, y_test = train_test_split(
        omics_data, target.to_frame(name='phenotype'), test_size=0.2, random_state=42
    )

    models = ["RawRidge", "GCN", "GAT", "SAGE", "GIN", "GraphTransformer"]
    results = {}

    for model_type in models:
        print(f"\nEvaluating {model_type} model...")
        if model_type == "RawRidge":
            try:
                reg = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
                reg.fit(X_train.values.astype(np.float64), y_train.values.flatten())
                y_pred = reg.predict(X_test.values.astype(np.float64))
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = float(np.sqrt(mse))
                emb_test = StandardScaler().fit_transform(X_test.values.astype(np.float64))
                var_emb_train = float(np.var(StandardScaler().fit_transform(X_train.values.astype(np.float64))))
                var_emb_test = float(np.var(emb_test))
                var_target = float(np.var(y_train.values.flatten()))
                var_pred = float(np.var(y_pred))
                results[model_type] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'embeddings': emb_test,
                    'y': y_test.values.flatten(),
                    'valid': np.isfinite(mse) and np.isfinite(r2) and var_emb_test > 1e-10 and var_target > 1e-10 and var_pred > 1e-10
                }
                print(f"{model_type} - RMSE: {rmse:.4f}, MSE: {mse:.4f}, R²: {r2:.4f} | var(emb_train)={var_emb_train:.6e}, var(emb_test)={var_emb_test:.6e}, var(y_train)={var_target:.6e}, var(y_pred)={var_pred:.6e}")
            except Exception as e:
                print(f"RawRidge baseline failed: {e}")
            continue

        if model_type == "GraphTransformer":
            gnn_embedding = GNNEmbedding(
                adjacency_matrix=adjacency_matrix,
                omics_data=X_train,
                phenotype_data=y_train,
                model_type=model_type,
                hidden_dim=64,
                layer_num=3,
                num_epochs=200,
                lr=5e-4,
                dropout=0.1,
                activation="gelu",
                weight_decay=0.0,
            )
        else:
            gnn_embedding = GNNEmbedding(
                adjacency_matrix=adjacency_matrix,
                omics_data=X_train,
                phenotype_data=y_train,
                model_type=model_type,
                hidden_dim=64,
                layer_num=2,
                num_epochs=50,
                lr=0.001,
                dropout=0.1
            )

        t0 = time.time()
        gnn_embedding.fit()

        node_embeddings = gnn_embedding.embed(as_df=False)
        t1 = time.time()

        nodes = adjacency_matrix.index.tolist()
        E = node_embeddings.detach().cpu().numpy().astype(np.float64)
        E = np.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)
        X_train_nodes = X_train[nodes].values.astype(np.float64)
        X_test_nodes = X_test[nodes].values.astype(np.float64)
        X_train_nodes = np.nan_to_num(X_train_nodes, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_nodes = np.nan_to_num(X_test_nodes, nan=0.0, posinf=1e6, neginf=-1e6)
        sample_embeddings_train = X_train_nodes.dot(E)
        sample_embeddings_test = X_test_nodes.dot(E)
        sample_embeddings_train = np.nan_to_num(sample_embeddings_train, nan=0.0, posinf=1e6, neginf=-1e6)
        sample_embeddings_test = np.nan_to_num(sample_embeddings_test, nan=0.0, posinf=1e6, neginf=-1e6)

        try:
            reg = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
            reg.fit(sample_embeddings_train, y_train.values.flatten())
            y_pred = reg.predict(sample_embeddings_test)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception as e:
            print(f"Ridge regression failed or unavailable ({e}); falling back to mean-based baseline.")
            y_pred = np.nan_to_num(sample_embeddings_test.mean(axis=1), nan=0.0, posinf=1e6, neginf=-1e-6)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        var_emb_train = float(np.var(sample_embeddings_train))
        var_emb_test = float(np.var(sample_embeddings_test))
        var_target = float(np.var(y_train.values.flatten()))
        var_pred = float(np.var(y_pred))

        results[model_type] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'embeddings': sample_embeddings_test,
                'y': y_test.values.flatten(),
                'valid': np.isfinite(mse) and np.isfinite(r2) and var_emb_test > 1e-10 and var_target > 1e-10 and var_pred > 1e-10,
                'train_time_s': float(t1 - t0)
            }

        print(f"{model_type} - RMSE: {rmse:.4f}, MSE: {mse:.4f}, R²: {r2:.4f} | var(emb_train)={var_emb_train:.6e}, var(emb_test)={var_emb_test:.6e}, var(y_train)={var_target:.6e}, var(y_pred)={var_pred:.6e}")
        if var_pred < 1e-10:
            print(f"Warning: {model_type} predictions are nearly constant; this often yields uniform metrics.")

        try:
            if isinstance(phenotype_data, (pd.Series, pd.DataFrame)) and not np.issubdtype(np.array(phenotype_data).dtype, np.number):
                y_cls_train = phenotype_data.loc[X_train.index]
                y_cls_test = phenotype_data.loc[X_test.index]
                if isinstance(y_cls_train, pd.DataFrame):
                    y_cls_train = y_cls_train.iloc[:, 0]
                if isinstance(y_cls_test, pd.DataFrame):
                    y_cls_test = y_cls_test.iloc[:, 0]
                le = LabelEncoder()
                yc_train = le.fit_transform(np.array(y_cls_train))
                yc_test = le.transform(np.array(y_cls_test))
                cls_pipe = Pipeline([
                    ('scale', StandardScaler()),
                    ('logreg', LogisticRegression(max_iter=200, multi_class='ovr'))
                ])
                cls_pipe.fit(sample_embeddings_train, yc_train)
                yc_pred = cls_pipe.predict(sample_embeddings_test)
                f1 = f1_score(yc_test, yc_pred, average='macro')
                try:
                    proba = cls_pipe.predict_proba(sample_embeddings_test)
                    if proba.shape[1] > 2:
                        auc = roc_auc_score(yc_test, proba, multi_class='ovr', average='macro')
                    else:
                        auc = roc_auc_score(yc_test, proba[:, 1])
                except Exception:
                    auc = float('nan')
                results[model_type].update({'f1': float(f1), 'auc': float(auc)})
        except Exception as cls_e:
            print(f"Classification evaluation failed for {model_type}: {cls_e}")

    plot_embeddings(results, models, VIZ_DIR)
    print("\nEvaluation complete!")
    plot_performance(results, models, VIZ_DIR)
    plot_classification_metrics(results, models, VIZ_DIR)

    try:
        from torch_geometric.utils import add_self_loops
        gnn_embedding = GNNEmbedding(
            adjacency_matrix=adjacency_matrix,
            omics_data=X_train,
            phenotype_data=y_train,
            model_type="GraphTransformer",
            hidden_dim=64,
            layer_num=2,
            num_epochs=10,
            lr=1e-3,
            dropout=0.1,
            activation="gelu",
            weight_decay=0.0,
        )
        gnn_embedding.fit()
        model = gnn_embedding.model
        if model is None:
            raise RuntimeError("Model not initialized; cannot visualize attention.")
        model.eval()
        _ = model(gnn_embedding.data)

        if hasattr(model, 'get_last_attentions'):
            atts = model.get_last_attentions()
            if atts and atts[0] and atts[0].get('alpha') is not None:
                alpha = atts[0]['alpha'].numpy()
                att_sum = alpha.mean(axis=1)
                cached_ei = atts[0].get('edge_index')
                if cached_ei is None:
                    raise RuntimeError("Cached edge_index not available; cannot align attentions to edges.")
                edges = cached_ei.numpy()
                num_nodes = int(edges.max()) + 1 if edges.size > 0 else adjacency_matrix.shape[0]
                heat = np.zeros((num_nodes, num_nodes), dtype=float)
                for e_i in range(edges.shape[1]):
                    src = int(edges[0, e_i])
                    dst = int(edges[1, e_i])
                    val = float(att_sum[e_i])
                    heat[dst, src] += val
                heat = np.clip(heat, 0.0, None)
                nz = int(np.count_nonzero(heat))
                total = int(heat.size)
                max_val = float(heat.max()) if nz > 0 else 0.0
                mean_val = float(heat.mean())
                print(f"Attention heat stats: nonzero={nz}/{total}, max={max_val:.6g}, mean={mean_val:.6g}")
                if nz > 0:
                    pos_vals = heat[heat > 0]
                    vmax = float(np.percentile(pos_vals, 99.0)) if pos_vals.size > 0 else float(heat.max())
                    vmin = 0.0
                else:
                    vmax = 1.0
                    vmin = 0.0
                SAVE_FULL_HEATMAP = bool(getattr(args, "full_heatmap", False) or getattr(args, "full_heatmap_force", False))
                MAX_N_FOR_FULL = int(getattr(args, "full_heatmap_max_n", 4096))
                FULL_DPI = int(getattr(args, "full_heatmap_dpi", 300))
                if getattr(args, "full_heatmap_force", False) or (SAVE_FULL_HEATMAP and num_nodes <= MAX_N_FOR_FULL):
                    plt.figure(figsize=(8, 6))
                    plt.imshow(heat, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                    plt.colorbar(label='Attention (avg across heads)')
                    plt.title('GraphTransformer Attention Heatmap (Layer 1)')
                    plt.xlabel('Source node')
                    plt.ylabel('Destination node')
                    plt.tight_layout()
                    plt.savefig(VIZ_DIR / 'graph_transformer_attention_heatmap.png', dpi=FULL_DPI)
                    print(f"Attention heatmap saved to '{VIZ_DIR / 'graph_transformer_attention_heatmap.png'}'")
                else:
                    print("Skipping full NxN attention heatmap due to scale; see binned/scatter/top-K images for global patterns.")
                try:
                    B = 256 if num_nodes >= 4096 else 128
                    bin_size = int(np.ceil(num_nodes / B))
                    H = np.zeros((B, B), dtype=float)
                    C = np.zeros((B, B), dtype=int)
                    for e_i in range(edges.shape[1]):
                        sb = int(edges[0, e_i] // bin_size)
                        db = int(edges[1, e_i] // bin_size)
                        if 0 <= sb < B and 0 <= db < B:
                            H[db, sb] += float(att_sum[e_i])
                            C[db, sb] += 1
                    H_avg = np.divide(H, C, out=np.zeros_like(H), where=C > 0)
                    nz_b = int(np.count_nonzero(H_avg))
                    vmax_b = float(np.percentile(H_avg[H_avg > 0], 99.0)) if nz_b > 0 else 1.0
                    plt.figure(figsize=(8, 6))
                    plt.imshow(H_avg, cmap='magma', aspect='auto', vmin=0.0, vmax=vmax_b)
                    plt.colorbar(label=f'Mean attention per bin (B={B})')
                    plt.title(f'GraphTransformer Attention Heatmap (Binned {B}x{B})')
                    plt.xlabel('Source node bin')
                    plt.ylabel('Destination node bin')
                    plt.tight_layout()
                    plt.savefig(VIZ_DIR / 'graph_transformer_attention_heatmap_binned.png', dpi=300)
                    print(f"Binned attention heatmap saved to '{VIZ_DIR / 'graph_transformer_attention_heatmap_binned.png'}'")
                except Exception as bin_e:
                    print(f"Could not generate binned attention heatmap: {bin_e}")
                try:
                    from matplotlib.colors import LogNorm
                    vmin_s = max(float(att_sum.min()), 1e-8)
                    vmax_s = float(att_sum.max()) if att_sum.size > 0 else 1.0
                    plt.figure(figsize=(8, 6))
                    plt.scatter(edges[0], edges[1], c=att_sum, s=0.2, cmap='viridis',
                                norm=LogNorm(vmin=vmin_s, vmax=vmax_s))
                    plt.xlim([0, num_nodes])
                    plt.ylim([0, num_nodes])
                    plt.xlabel('Source node')
                    plt.ylabel('Destination node')
                    plt.title('GraphTransformer Attention (Edge Scatter, log-scale)')
                    plt.tight_layout()
                    plt.savefig(VIZ_DIR / 'graph_transformer_attention_edges_scatter.png', dpi=300)
                    print(f"Edge scatter attention plot saved to '{VIZ_DIR / 'graph_transformer_attention_edges_scatter.png'}'")
                except Exception as scat_e:
                    print(f"Could not generate edge scatter attention plot: {scat_e}")
                try:
                    k = min(200, edges.shape[1])
                    top_idx = np.argsort(att_sum)[-k:]
                    top_nodes = np.unique(edges[:, top_idx])
                    sub_heat = heat[np.ix_(top_nodes, top_nodes)]
                    sub_vmax = float(np.percentile(sub_heat[sub_heat > 0], 99.0)) if np.count_nonzero(sub_heat) > 0 else 1.0
                    plt.figure(figsize=(8, 6))
                    plt.imshow(sub_heat, cmap='inferno', aspect='auto', vmin=0.0, vmax=sub_vmax)
                    plt.colorbar(label='Attention (avg across heads)')
                    plt.title('GraphTransformer Attention Heatmap (Top-K Subgraph)')
                    plt.xlabel('Source node (subset)')
                    plt.ylabel('Destination node (subset)')
                    plt.tight_layout()
                    plt.savefig(VIZ_DIR / 'graph_transformer_attention_heatmap_topk.png', dpi=300)
                    print(f"Top-K attention subgraph heatmap saved to '{VIZ_DIR / 'graph_transformer_attention_heatmap_topk.png'}'")
                except Exception as sub_e:
                    print(f"Could not generate top-K subgraph heatmap: {sub_e}")
            else:
                print("Attention weights not available for visualization.")
        else:
            print("Model does not expose attention weights.")
    except Exception as att_e:
        print(f"Could not generate attention visualization: {att_e}")

if __name__ == "__main__":
    run_evaluation_cli()
