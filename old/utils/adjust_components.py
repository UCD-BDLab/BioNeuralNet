# Bioneuralnet/utils/adjust_components.py

import os
import yaml
import logging

# mapping from component number to component name for config file resolution
component_mapping = {
    1: "graph_generation",
    2: "clustering",
    3: "network_embedding",
    4: "subject_representation",
    5: "integrated_tasks"
}

# second mapping from each component number to its run function
# set externally in main.py
component_run_functions = {}

def adjust_component_config(component_number, root_config, previous_output_dir=None):
    component_name = component_mapping[component_number]
    component_config_path = os.path.join(f"m{component_number}_{component_name}", "config", "config.yml")

    if not os.path.isfile(component_config_path):
        logging.error(f"Component config file not found: {component_config_path}")
        raise FileNotFoundError(f"Component config file not found: {component_config_path}")

    # Loading component specific configuration file.
    with open(component_config_path, 'r') as file:
        component_config = yaml.safe_load(file)

    # checking the current working directory and component config path
    cwd = os.getcwd()
    logging.debug(f"Current working directory: {cwd}")
    logging.debug(f"Component config path: {component_config_path}")

    # now the dir of the component config file
    component_config_dir = os.path.dirname(os.path.abspath(component_config_path))
    logging.debug(f"Component config directory: {component_config_dir}")

    # modifying the input_dir and output_dir paths
    if previous_output_dir:
        input_dir = previous_output_dir
    else:
        input_dir = component_config[component_name]['paths']['input_dir']
        if not os.path.isabs(input_dir):
            input_dir = os.path.abspath(os.path.join(component_config_dir, input_dir))
        component_config[component_name]['paths']['input_dir'] = input_dir

    logging.debug(f"Resolved input directory: {input_dir}")

    # making sure input_dir exists
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # doing the same for output_dir
    output_dir = component_config[component_name]['paths']['output_dir']
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(component_config_dir, output_dir))
    component_config[component_name]['paths']['output_dir'] = output_dir

    logging.debug(f"Resolved output directory: {output_dir}")

    return component_config


def run_component(component_number, component_config):
    """
    Run the specified component with its configuration.
    """
    run_function = component_run_functions.get(component_number)
    if not run_function:
        raise ValueError(f"No run function defined for component {component_number}.")
    run_function(component_config)
