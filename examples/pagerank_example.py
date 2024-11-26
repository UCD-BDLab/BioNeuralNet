from bioneuralnet.clustering.pagerank import PageRankClustering

def main():
    # Initialize PageRankClustering parameters
    pagerank_cluster = PageRankClustering(
        graph_file='input/GFEV1ac110.edgelist',
        omics_data_file='input/X.xlsx',
        phenotype_data_file='input/Y.xlsx',
        alpha=0.9,
        max_iter=100,
        tol=1e-6,
        k=0.9,
    )

    # Define seed nodes 
    seed_nodes = [94] 

    # Run PageRank clustering
    results = pagerank_cluster.run(seed_nodes=seed_nodes)

    # Access results
    cluster_nodes = results['cluster_nodes']
    print(f"Identified cluster with {len(cluster_nodes)} nodes.")

if __name__ == "__main__":
    main()
