all_tasks = ["prediction", "downstream_task"]
all_algorithms = ["random_forest", "downstreamtask_algorithm"]


def str2task(task_name, algorithm_name):
    if task_name == "prediction":
        if algorithm_name == "random_forest":
            from .config.prediction import run_method
            return run_method
    elif task_name == "downstream_task":
        if algorithm_name == "downstreamtask_algorithm":
            from .config.downstream import run_method
            return run_method
    else:
        raise ValueError(f"Task '{task_name}' with `{algorithm_name}` not recognized in task_optimization.")
