# all_tasks = ["prediction", "downstream_task"]

# def str2task(task_name):
#     """
#     Maps a task name to its corresponding function.

#     Args:
#         task_name (str): The name of the task to execute.

#     Returns:
#         function: The function that executes the specified task.

#     Raises:
#         ValueError: If the task name is not recognized.
#     """
#     if task_name == "prediction":
#         from .DPMON import run_prediction_dpmon
#         return run_prediction_dpmon

#     #Just a placeholder for now
#     elif task_name == "downstream_task":
#         from .run import run_downstream_task
#         return run_downstream_task
#     else:
#         raise ValueError(f"Task '{task_name}' not recognized in integrated_tasks.")

# # Function to list available tasks
# def list_available_tasks():
#     """
#     Returns a list of all available tasks.

#     Returns:
#         list: List of task names.
#     """
#     return all_tasks
