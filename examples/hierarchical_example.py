from bioneuralnet.clustering.hierarchical import HierarchicalClustering

def main():
    # Initialize HierarchicalClustering parameters
    hierarchical_cluster = HierarchicalClustering(
        adjacency_matrix_file='input/global_network.csv',
        n_clusters=3,            
        linkage='ward',         
        affinity='euclidean',   
    )

    # Run the hierarchical clustering
    results = hierarchical_cluster.run()

    # Access results
    cluster_labels_df = results['cluster_labels']
    print("Cluster labels:")
    print(cluster_labels_df.head())

    silhouette_score = results['silhouette_score']
    print(f"Silhouette Score: {silhouette_score}")

if __name__ == "__main__":
    main()
