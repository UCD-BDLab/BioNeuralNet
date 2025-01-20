from bioneuralnet.clustering import PageRank
import networkx as nx
import pandas as pd


def main():
    graph = nx.read_edgelist("input/GFEV1ac110.edgelist", data=(("weight", float),))
    omics = pd.read_csv("Input/X.csv").iloc[:, 1:]
    phenotype = pd.read_csv("Input/Y.csv").iloc[:, 1]

    omics.columns = [str(index) for index in range(omics.shape[1])]

    pagerank_cluster = PageRank(
        graph=graph,
        omics_data=omics,
        phenotype_data=phenotype,
        alpha=0.9,
        max_iter=100,
        tol=1e-6,
        k=0.9,
    )

    seed_nodes = [
        "251",
        "325",
        "303",
        "358",
        "445",
        "636",
        "374",
        "96",
        "1159",
        "324",
        "2221",
        "1884",
        "1985",
    ]
    seed_nodes = [
        "251",
        "325",
        "303",
        "358",
        "445",
        "636",
        "374",
        "96",
        "1159",
        "324",
        "2221",
        "1884",
        "1985",
    ]
    results = pagerank_cluster.run(seed_nodes=seed_nodes)
    cluster_nodes = results["cluster_nodes"]
    print(f"Identified cluster with {len(cluster_nodes)} nodes: {cluster_nodes}")


if __name__ == "__main__":
    main()
