all_algorithms = ["smccnet", "MGCNA"]

def str2algo(algorithm_name):
    if algorithm_name == "smccnet":
        from .config.smccnet import run_smccnet
        return run_smccnet
    elif algorithm_name == "MGCNA":
        from .config.MGCNA import run_mgcna
        return run_mgcna
    else:
        raise ValueError(f"Algorithm '{algorithm_name}' not recognized in graph_generation.")
