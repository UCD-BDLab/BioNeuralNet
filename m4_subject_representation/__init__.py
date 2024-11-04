all_algorithms = ["concatenate", "summarize"]

def str2algo(algorithm_name):
    if algorithm_name == "concatenate":
        from .config.concatenate import run_method
        return run_method
    elif algorithm_name == "summarize":
        from .config.summarize import run_method
        return run_method
    else:
        raise ValueError(f"Method '{algorithm_name}' not recognized in subject_representation.")
