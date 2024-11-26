all_algorithms = ["PageRank", "Hierarchical"]

def str2algo(algorithm_name):
    if algorithm_name == "PageRank":
        from .config.PageRank import run_pagerank
        return run_pagerank
    elif algorithm_name == "Hierarchical":
        from .config.Hierarchical import run_hierarchical
        return run_hierarchical
    else:
        raise ValueError(f"Algorithm '{algorithm_name}' not recognized in clustering.")
