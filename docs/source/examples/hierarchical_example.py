import pandas as pd
from bioneuralnet.external_tools import HierarchicalClustering


def main():
    try:
        print("Starting Hierarchical Clustering Workflow...")
        adjacency_matrix = pd.DataFrame(
            {
                "GeneA": [1.0, 0.8, 0.3, 0.2],
                "GeneB": [0.8, 1.0, 0.4, 0.3],
                "GeneC": [0.3, 0.4, 1.0, 0.5],
                "GeneD": [0.2, 0.3, 0.5, 1.0],
            },
            index=["GeneA", "GeneB", "GeneC", "GeneD"],
        )

        hierarchical_clustering = HierarchicalClustering(
            adjacency_matrix=adjacency_matrix,
            n_clusters=2,
            linkage="ward",
            affinity="euclidean",
            scale_data=True,
        )

        clustering_results = hierarchical_clustering.run()

        print("\nCluster Labels:")
        print(clustering_results["cluster_labels"])

        silhouette_score = clustering_results["silhouette_score"]
        if silhouette_score is not None:
            print(f"\nSilhouette Score: {silhouette_score:.4f}")
        else:
            print("\nSilhouette Score could not be computed.")

        print("\nHierarchical Clustering Workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


if __name__ == "__main__":
    main()
