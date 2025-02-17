import networkx as nx
import pandas as pd
from bioneuralnet.datasets import DatasetLoader
from bioneuralnet.external_tools import SmCCNet
from bioneuralnet.clustering import HybridLouvain
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

# Perform Hybrid Louvain Clustering
hybrid_louvain_instance = HybridLouvain(
    G=G_network,
    B=merged_omics,
    Y=phenotype,
    k3=0.2,
    k4=0.8,
    max_iter=10,  # Number of refinement iterations
    weight="weight",
    tune=True
)
hybrid_louvain_results = hybrid_louvain_instance.run()

# Extract final partitions
final_clusters = hybrid_louvain_results["curr"]
iterative_clusters = hybrid_louvain_results["clus"]

print(f"Final Hybrid Louvain Clusters: {len(set(final_clusters.values()))}")
print(f"Number of Iterative Refinements: {len(iterative_clusters)}")

# Compare clusters against SmCCNet clusters
compare_clusters(final_clusters, smccnet_clusters, phenotype)
