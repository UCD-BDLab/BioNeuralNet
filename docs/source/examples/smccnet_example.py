import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet

# Load example synthetic dataset
loader = DatasetLoader("example1")
omics1, omics2, phenotype, clinical = loader.load_data()

# Display dataset dimensions
print("Dataset Shapes:")
print(f"Omics1: {omics1.shape}")  # Expected: (358, 500)
print(f"Omics2: {omics2.shape}")  # Expected: (358, 100)
print(f"Phenotype: {phenotype.shape}")  # Expected: (358, 1)
print(f"Clinical: {clinical.shape}")  # Expected: (358, 6)")

# Merge omics and clinical data
merged_omics = pd.concat([omics1, omics2, clinical, phenotype], axis=1)

# Initialize and run SmCCNet
smccnet = SmCCNet(
    phenotype_df=phenotype,
    omics_dfs=[omics1, omics2],
    data_types=["genes", "proteins"],
    kfold=3,
    subSampNum=500,
)

global_network, smccnet_clusters = smccnet.run()

# Display output sizes
print(f"Global Network Shape: {global_network.shape}")
print(f"Number of SmCCNet Clusters: {len(smccnet_clusters)}")
