import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.network_embedding import GNNEmbedding
from bioneuralnet.subject_representation import GraphEmbedding

# Load example synthetic dataset
loader = DatasetLoader("example1")
omics1, omics2, phenotype, clinical = loader.load_data()

# Display dataset dimensions
print("Dataset Shapes:")
print(f"Omics1: {omics1.shape}")  # Expected: (358, 500)
print(f"Omics2: {omics2.shape}")  # Expected: (358, 100)
print(f"Phenotype: {phenotype.shape}")  # Expected: (358, 1)
print(f"Clinical: {clinical.shape}")  # Expected: (358, 6)")

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

# Display output shape and preview
print(f"Enhanced Omics Shape: {enhanced_omics_df.shape}")
print(enhanced_omics_df.head())
