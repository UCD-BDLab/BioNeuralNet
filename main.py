# main.py

import os
import sys
import yaml
import logging
from utils import setup_logging
from utils import get_parser
from utils import adjust_component_config, run_component, component_run_functions, component_mapping

# Import component modules with numerical prefixes
import m1_graph_generation.run as graph_generation
import m2_clustering.run as clustering
import m3_network_embedding.run as network_embedding
import m4_subject_representation.run as subject_representation
import m5_task_optimization.run as task_optimization

# Assign run functions to component_run_functions mapping
component_run_functions[1] = graph_generation.run_graph_generation
component_run_functions[2] = clustering.run_clustering
component_run_functions[3] = network_embedding.run_network_embedding
component_run_functions[4] = subject_representation.run_subject_representation
component_run_functions[5] = task_optimization.run_task_optimization

def load_root_config(config_path='config.yml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def display_menu():
    print("BioNeuralNet Interface")
    print("Select the components you want to run:")
    print("1. Graph Generation")
    print("2. Clustering")
    print("3. Network Embedding")
    print("4. Subject Representation")
    print("5. Task Optimization")
    #print("No specified method selected. Using defaults.")
    print("Enter the numbers of the components separated by commas (e.g., 1,3):")

def main():
    # If no command-line arguments are provided, display the menu
    if len(sys.argv) == 1:
        display_menu()
        user_input = input().strip()
        selected_components = [int(x) for x in user_input.split(',') if x.strip().isdigit()]

        # Load root config
        root_config = load_root_config()

        # Set up root logging
        output_dir = root_config['pipeline'].get('output_dir', 'global_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pipeline_log = os.path.join(output_dir, root_config['pipeline'].get('log_file', 'pipeline.log'))
        setup_logging(pipeline_log)
        logger = logging.getLogger(__name__)

        logger.info("Pipeline configuration loaded from config.yml")

        # Validate selected components
        valid_components = [1, 2, 3]
        for comp in selected_components:
            if comp not in valid_components:
                print(f"Component {comp} is either invalid or under construction.")
                return

        # Run the selected components
        run_selected_components(selected_components, root_config, logger)

    else:
        # Existing functionality using --start and --end
        parser = get_parser()
        args = parser.parse_args()

        # Load root config
        root_config = load_root_config()

        # Set up root logging
        output_dir = root_config['pipeline'].get('output_dir', 'global_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pipeline_log = os.path.join(output_dir, root_config['pipeline'].get('log_file', 'pipeline.log'))
        setup_logging(pipeline_log)
        logger = logging.getLogger(__name__)

        logger.info("Pipeline configuration loaded from config.yml")

        run_all = root_config['pipeline'].get('run_all', False)

        start = args.start
        end = args.end

        logger.info(f"Pipeline run_all: {run_all}")
        logger.info(f"Pipeline execution range: {start} to {end}")

        if start < 1 or start > 5 or end < 1 or end > 5:
            logger.error("Component indices must be between 1 and 5.")
            sys.exit(1)

        if start > end:
            logger.error("--start cannot be greater than --end.")
            sys.exit(1)

        selected_components = list(range(start, end + 1))
        run_selected_components(selected_components, root_config, logger)

def run_selected_components(selected_components, root_config, logger):
    previous_output_dir = None

    for component_number in selected_components:
        component_name = component_mapping[component_number]
        logger.info(f"Running component {component_number}: {component_name}")

        if component_number > 3:
            print(f"Component {component_number} ({component_name}) is under construction.")
            continue

        # Adjust config based on previous component's output if necessary
        if previous_output_dir:
            component_config = adjust_component_config(component_number, root_config, previous_output_dir)
        else:
            component_config = adjust_component_config(component_number, root_config)

        # Run the component
        run_component(component_number, component_config)

        # Update previous_output_dir for the next component
        output_dir = component_config[component_name]['paths']['output_dir']
        if output_dir:
            previous_output_dir = os.path.abspath(output_dir)
            logger.info(f"Component {component_number} completed. Outputs saved to {previous_output_dir}")
        else:
            logger.warning(f"No output_dir found for component {component_number}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"An error occurred: {e}")
        sys.exit(1)