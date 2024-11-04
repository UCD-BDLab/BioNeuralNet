all_algorithms = ["node2vec", "GNN"]

def str2algo(algorithm_name):
    if algorithm_name == "node2vec":
        from .config.Node2Vec import run_node2vec
        return run_node2vec
    elif algorithm_name == "GNN":
        from .config.GNNs import run_gnns
        return run_gnns
    else:
        raise ValueError(f"Algorithm '{algorithm_name}' not recognized in network_embedding.")
