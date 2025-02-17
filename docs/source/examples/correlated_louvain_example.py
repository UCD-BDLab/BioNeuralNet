import networkx as nx
import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.clustering import CorrelatedLouvain
from bioneuralnet.utils import compare_clusters

# Load example synthetic dataset
loader = DatasetLoader("example1")
omics1, omics2, phenotype, clinical = loader.load_data()

# Display dataset dimensions
print("Dataset Shapes:")
print(f"Omics1: {omics1.shape}")  # Expected: (358, 500)
print(f"Omics2: {omics2.shape}")  # Expected: (358, 100)
print(f"Phenotype: {phenotype.shape}")  # Expected: (358, 1)
print(f"Clinical: {clinical.shape}")  # Expected: (358, 6)")

# Generate global network using SmCCNet
smccnet = SmCCNet(
    phenotype_df=phenotype,
    omics_dfs=[omics1, omics2],
    data_types=["genes", "proteins"],
    kfold=3,
    subSampNum=500,
)
global_network, smccnet_clusters = smccnet.run()

# Convert adjacency matrix to NetworkX graph
merged_omics = pd.concat([omics1, omics2], axis=1)
G_network = nx.from_pandas_adjacency(global_network)

# Perform Correlated Louvain Clustering
louvain_instance = CorrelatedLouvain(
    G=G_network,
    B=merged_omics,
    Y=phenotype,
    k3=0.2,
    k4=0.8,
    weight="weight",
    tune=True
)
louvain_clusters = louvain_instance.run(as_dfs=True)

# Compare clusters against SmCCNet clusters
print(f"Number of Louvain Clusters: {len(louvain_clusters)}")
compare_clusters(louvain_clusters, smccnet_clusters, phenotype)
