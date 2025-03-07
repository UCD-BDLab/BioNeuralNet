import numpy as np
import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.network_embedding import GNNEmbedding
from bioneuralnet.subject_representation import GraphEmbedding
from bioneuralnet.downstream_task import DPMON
from bioneuralnet.metrics import evaluate_rf, plot_performance

# Load example synthetic dataset
loader = DatasetLoader("example1")
omics1, omics2, phenotype, clinical = loader.load_data()

# Display dataset dimensions
print("Dataset Shapes:")
print(f"Omics1: {omics1.shape}")  # Expected: (358, 500)
print(f"Omics2: {omics2.shape}")  # Expected: (358, 100)
print(f"Phenotype: {phenotype.shape}")  # Expected: (358, 1)
print(f"Clinical: {clinical.shape}")  # Expected: (358, 6)")

# Preprocess phenotype data: Convert continuous values into discrete bins
min_val = phenotype["phenotype"].min()
max_val = phenotype["phenotype"].max()
bins = np.linspace(min_val, max_val, 5)  # Creates 4 categories
phenotype["phenotype"] = pd.cut(phenotype["phenotype"], bins=bins, labels=[0, 1, 2, 3], include_lowest=True)

print("Binned Phenotype Data:")
print(phenotype.head())
print(phenotype["phenotype"].value_counts(sort=False))

# Merge omics data
merged_omics = pd.concat([omics1, omics2], axis=1)

# Generate global network using SmCCNet
smccnet = SmCCNet(
    phenotype_df=phenotype,
    omics_dfs=[omics1, omics2],
    data_types=["genes", "proteins"],
    kfold=3,
    subSampNum=500,
)
global_network, smccnet_clusters = smccnet.run()

# Generate node embeddings using GNNEmbedding
embeddings = GNNEmbedding(
    adjacency_matrix=global_network,
    omics_data=merged_omics,
    phenotype_data=phenotype,
    clinical_data=clinical,
    tune=True,
)
embeddings.fit()
embeddings_output = embeddings.embed(as_df=True)

# Perform Subject Representation using GraphEmbedding
enhanced_omics = GraphEmbedding(
    omics_data=merged_omics,
    embeddings=embeddings_output,
    phenotype_data=phenotype,
    tune=True,
)
enhanced_omics_df = enhanced_omics.run()

print(f"Enhanced Omics Shape: {enhanced_omics_df.shape}")

# Run Disease Prediction using DPMON
dpmon = DPMON(
    adjacency_matrix=global_network,
    omics_list=[omics1, omics2],
    phenotype_data=phenotype,
    clinical_data=clinical,
    tune=True,
)

dpmon_predictions = dpmon.run()
print(f"DPMON Predictions:\n{dpmon_predictions[0]}")

# Evaluate Classifier Performance
X_raw = merged_omics.values
y_global = phenotype.values
raw_rf_acc = evaluate_rf(X_raw, y_global, mode='classification')

print("Global Results:")
plot_performance(dpmon_predictions[1], raw_rf_acc, "Raw Omics vs. DPMON Omics")
