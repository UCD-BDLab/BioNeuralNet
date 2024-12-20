from bioneuralnet.clustering import PageRank

def main():
    pagerank_cluster = PageRank(
        graph_file='input/GFEV1ac110.edgelist',
        omics_data_file='input/X.xlsx',
        phenotype_data_file='input/Y.xlsx',
        alpha=0.9,
        max_iter=100,
        tol=1e-6,
        k=0.9,
    )

    seed_nodes = [94] 
    results = pagerank_cluster.run(seed_nodes=seed_nodes)
    cluster_nodes = results['cluster_nodes']
    print(f"Identified cluster with {len(cluster_nodes)} nodes.")

    # Save results
    results.to_csv('output/pagerank_results.csv')
    print("PageRank clustering results saved to 'output/pagerank_results.csv'.")

if __name__ == "__main__":
    main()
