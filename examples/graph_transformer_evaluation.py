"""
Graph Transformer Evaluation Script

This script demonstrates how to use the Graph Transformer model in BioNeuralNet
and compares its performance with traditional GNN models (GCN, GAT, SAGE, GIN)
on a sample dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
from pathlib import Path

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

# Load a sample dataset (BRCA dataset)
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

    # Generating synthetic omics data
    omics_data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Generating synthetic phenotype data
    true_weights = np.random.normal(0, 1, size=n_features)
    phenotype_data = pd.DataFrame({
        'survival_time': np.dot(omics_data, true_weights) + np.random.normal(0, 0.5, size=n_samples)
    })

# Cleaning and preparing data
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

# Generating a KNN graph from the omics data
adjacency_matrix = gen_gaussian_knn_graph(omics_data, k=20)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    omics_data, target.to_frame(name='phenotype'), test_size=0.2, random_state=42
)

# Defining GNN models to compare
models = ["RawRidge", "GCN", "GAT", "SAGE", "GIN", "GraphTransformer"]
results = {}

# Training and evaluate each model
for model_type in models:
    print(f"\nEvaluating {model_type} model...")
    # Baseline using raw features with Ridge regression
    if model_type == "RawRidge":
        try:
            reg = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
            reg.fit(X_train.values.astype(np.float64), y_train.values.flatten())
            y_pred = reg.predict(X_test.values.astype(np.float64))
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            # Using standardized raw features as embeddings for visualization
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

    # Training the model
    gnn_embedding.fit()

    # Generating node embeddings (for graph nodes/features)
    node_embeddings = gnn_embedding.embed(as_df=False)

    nodes = adjacency_matrix.index.tolist()
    E = node_embeddings.detach().cpu().numpy().astype(np.float64)  # [num_nodes, hidden_dim]
    E = np.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)
    X_train_nodes = X_train[nodes].values.astype(np.float64)  # [n_train, num_nodes]
    X_test_nodes = X_test[nodes].values.astype(np.float64)    # [n_test, num_nodes]
    X_train_nodes = np.nan_to_num(X_train_nodes, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_nodes = np.nan_to_num(X_test_nodes, nan=0.0, posinf=1e6, neginf=-1e6)
    sample_embeddings_train = X_train_nodes.dot(E)  # [n_train, hidden_dim]
    sample_embeddings_test = X_test_nodes.dot(E)    # [n_test, hidden_dim]
    sample_embeddings_train = np.nan_to_num(sample_embeddings_train, nan=0.0, posinf=1e6, neginf=-1e6)
    sample_embeddings_test = np.nan_to_num(sample_embeddings_test, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        reg = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        reg.fit(sample_embeddings_train, y_train.values.flatten())
        y_pred = reg.predict(sample_embeddings_test)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception as e:
        print(f"Ridge regression failed or unavailable ({e}); falling back to mean-based baseline.")
        y_pred = np.nan_to_num(sample_embeddings_test.mean(axis=1), nan=0.0, posinf=1e6, neginf=-1e6)

    # Calculating evaluation metrics
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
            'valid': np.isfinite(mse) and np.isfinite(r2) and var_emb_test > 1e-10 and var_target > 1e-10 and var_pred > 1e-10
        }

    print(f"{model_type} - RMSE: {rmse:.4f}, MSE: {mse:.4f}, R²: {r2:.4f} | var(emb_train)={var_emb_train:.6e}, var(emb_test)={var_emb_test:.6e}, var(y_train)={var_target:.6e}, var(y_pred)={var_pred:.6e}")
    if var_pred < 1e-10:
        print(f"Warning: {model_type} predictions are nearly constant; this often yields uniform metrics.")

# Visualizing embeddings using t-SNE
try:
    from sklearn.manifold import TSNE
    import seaborn as sns
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
            if n_samples > max_points:
                idx = np.random.RandomState(42).choice(n_samples, size=max_points, replace=False)
                emb_plot = emb[idx]
                y_plot = results[model_type]['y'][idx]
            else:
                emb_plot = emb
                y_plot = results[model_type]['y'][:emb.shape[0]]
            # Degeneracy checks
            col_std = emb_plot.std(axis=0)
            row_std = emb_plot.std(axis=1)
            degenerate = (np.all(col_std < 1e-12)) or (np.all(row_std < 1e-12)) or (float(np.var(emb_plot)) < 1e-12)
            plt.subplot(2, 3, i+1)
            if degenerate or emb_plot.shape[0] < 10:
                # PCA fallback or annotation
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
    plt.savefig(VIZ_DIR / 'embedding_visualizations.png', dpi=300)
    print(f"Embedding visualizations saved to '{VIZ_DIR / 'embedding_visualizations.png'}'")
except Exception as e:
    print(f"Could not generate embedding visualizations: {e}")

print("\nEvaluation complete!")

# Creating and saving comparison plot after evaluation loop
try:
    # VIZ_DIR already created above

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
        # Ensure at least RawRidge is shown if available
        if 'RawRidge' in results:
            plt.bar([0], [results['RawRidge']['r2']], color='steelblue', edgecolor='black')
            plt.xticks([0], ['RawRidge'])
            plt.ylabel('R² score')
            plt.title('Graph Transformer Comparison (R²)')
        else:
            plt.text(0.5, 0.5, "No valid model results to plot", ha='center', va='center')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'graph_transformer_comparison.png', dpi=300)
    print(f"Comparison plot saved to '{VIZ_DIR / 'graph_transformer_comparison.png'}'")
except Exception as plot_e:
    print(f"Could not generate graph_transformer_comparison plot: {plot_e}")
