import re
import torch
import os.path
import numpy as np
import pandas as pd
import networkx as nx
import torch.utils.data
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import logging


# module level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_omics_dataset(phenotype_file, omics_files):
    """
    Load and merge omics data files with the phenotype file.

    Args:
        omics_files (list of str): Paths to omics data CSV files.
        phenotype_file (str): Path to the phenotype CSV file.

    Returns:
        pd.DataFrame: Merged omics and phenotype data.
    """
    logger.debug(f"Loading phenotype file from: {phenotype_file}")
    logger.debug(f"Loading omics files from: {omics_files}")

    omics_datasets = []
    for omics_file in omics_files:
        omics_df = pd.read_csv(omics_file, index_col=0)
        omics_datasets.append(omics_df)

    # Merge all omics datasets on their index
    omics_dataset_complete = pd.concat(omics_datasets, axis=1)

    phenotype = pd.read_csv(phenotype_file, index_col=0)
    omics_dataset_complete = omics_dataset_complete.merge(phenotype, how='left', left_index=True, right_index=True)

    # Replace PRISM (-1) with 5 for Prediction to Work!
    omics_dataset_complete['finalgold_visit'] = np.where(
        omics_dataset_complete['finalgold_visit'] == -1, 5, omics_dataset_complete['finalgold_visit']
    )

    logger.info(f"Final Gold: {omics_dataset_complete['finalgold_visit'].value_counts}")
    omics_dataset_complete = omics_dataset_complete.dropna()

    logger.debug(f"Finished combining omics and phenotype data. Shape: {omics_dataset_complete.shape}")

    return omics_dataset_complete


def get_omics_datasets(omics_dataset, network_file):
    """
    Slice the omics dataset based on the nodes in the network file.
    Mostly used for clustered networks.

    Args:
        omics_dataset (pd.DataFrame): Complete omics dataset with phenotype.
        network_file (str): Path to the network CSV file.

    Returns:
        list of pd.DataFrame: Sliced omics datasets for each network.
    """

    # Load the network adjacency matrix
    omics_network_adj = pd.read_csv(network_file, index_col=0)
    logger.info(f"Omics Dataset shape: {omics_dataset.shape}")
    logger.info(f"Network File shape: {omics_network_adj.shape}")


    omics_datasets = []
    omics_network_nodes_names = omics_network_adj.index.tolist()

    # SmCCnet will replace non-alphanumeric characters except underscores with '.' and prefix with 'X' if it does not start with a letter
    # Here we are doing the same to match the columns in the omics dataset
    clean_columns = []
    for node in omics_dataset:
        node_clean = re.sub(r'[^0-9a-zA-Z_]', '.', node)
        if not node_clean[0].isalpha():
            node_clean = 'X' + node_clean
        clean_columns.append(node_clean)

    #logger.info(f"Cleaned columns: {clean_columns}")
    omics_dataset.columns = clean_columns

    # Ensure that all nodes exist in omics_dataset **columns**
    missing_nodes = set(omics_network_nodes_names) - set(omics_dataset.columns.tolist())
    if missing_nodes:
        logger.info(f"Total Number of Nodes Missing: {len(missing_nodes)}")
        logger.info(f"Nodes Missing: {missing_nodes}")
        raise ValueError(f"Nodes Missing, check component logs.")
    
    # Slice the omics dataset based on the nodes in the network file + phenotype
    selected_columns = omics_network_nodes_names + [omics_dataset.columns[-1]]
    omics_datasets.append(omics_dataset[selected_columns])

    logger.info(f"Omics Datasets shape: {len(omics_datasets)}")
    logger.info(f"Network File shape: {omics_network_adj.shape[1]}")
    logger.info(f"Checking if slicing was succesfull succesful: {len(omics_datasets) == omics_network_adj.shape[1]}")

    return omics_datasets



def get_omics_networks_tg(network_file, omics_dataset, features_file=None):
    """
    Load the network file and create PyTorch Geometric Data objects.
    If features_file is provided, compute node features based on correlations
    between the omics dataset and features (clinical data). Otherwise, generate
    random node features.

    Args:
        network_file (str): Path to the network CSV file.
        omics_dataset (pd.DataFrame): Merged omics and phenotype dataset.
        features_file (str, optional): Path to the features CSV file.

    Returns:
        list of Data: List containing a PyTorch Geometric Data object for the network.
    """
    logger.debug(f"Loading network file from: {network_file}")
    logger.debug(f"Loading features file from: {features_file}")

    # Load network adjacency matrix
    omics_network_adj = pd.read_csv(network_file, index_col=0)
    omics_network_nodes_names = omics_network_adj.index.tolist()

    # Create NetworkX graph from adjacency matrix
    omics_network = nx.from_pandas_adjacency(omics_network_adj)

    # Creating a dictionary to map node names to integer indices
    node_mapping = {}
    for idx, node_name in enumerate(omics_network_nodes_names):
        node_mapping[node_name] = idx
   
    # Relabel the nodes in the graph to integers
    omics_network = nx.relabel_nodes(omics_network, node_mapping)

    # Number of nodes: for debugging purposes
    num_nodes = len(node_mapping)
    logger.info(f"Number of nodes in network: {num_nodes}")

    # Load node features
    if features_file and os.path.isfile(features_file):
        # Load features (clinical dataset)
        # We need to test this part since I did not have the clinical dataset
        features_df = pd.read_csv(features_file, index_col=0)
        if not features_df.empty:

            # Define clinical variables to use for correlations
            clinical_vars = features_df.columns.tolist()
            logger.info(f"Using clinical variables: {clinical_vars}")

            # Ensure omics_dataset contains necessary nodes
            # Thinking of removing this since we are already checking in get_omics_datasets
            missing_nodes = set(omics_network_nodes_names) - set(omics_dataset.columns)
            if missing_nodes:
                raise ValueError(f"Nodes {missing_nodes} in the network are missing in omics_dataset.")

            # Compute node features based on correlations
            node_features = []
            for node_name in omics_network_nodes_names:
                correlations = []
                for var in clinical_vars:
                    # Computing absolute correlation
                    corr_value = abs(omics_dataset[node_name].corr(features_df[var].astype('float64')))
                    correlations.append(corr_value)

                node_features.append(correlations)

            x = torch.tensor(node_features, dtype=torch.float)
            logger.info(f"Computed node features based on correlations.")
        else:
            # Generate random features if features_file is empty
            # might remove this part
            x = torch.randn((num_nodes, 10), dtype=torch.float)
            logger.info(f"{features_file} is empty. Generated random node features.")
    else:
        # Generate random features if features_file is not provided or does not exist
        # Currently using this
        x = torch.randn((num_nodes, 10), dtype=torch.float)
        logger.info("No features_file provided. Generated random node features.")

    # now we get the index of the nodes in the network and duplicate them for undirected graph

    # Edge indices
    edge_index = torch.tensor(list(omics_network.edges()), dtype=torch.long).t().contiguous()

    # Duplicate edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Edge weights
    edge_weight = torch.tensor([
        data.get('weight', 1.0) for _, _, data in omics_network.edges(data=True)
    ], dtype=torch.float)

    # Duplicate edge weights
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)


    # lastly well create the PyTorch Geometric Data object to be used in the model
    # Create PyTorch Geometric Data object

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    # Define validation and test splits
    num_nodes = len(omics_network.nodes())
    num_val_nodes = 10
    num_test_nodes = 12
    if num_nodes < 30:
        num_val_nodes = 5
        num_test_nodes = 5

    transform = RandomNodeSplit(num_val=num_val_nodes, num_test=num_test_nodes)
    data = transform(data)

    return [data]

def get_dataset(dpmon_params):
    """
    Main driver for dataset loading.

    Args:
        dpmon_params (dict): Dictionary containing 'phenotype_file', 'omics_files', 'network_file', and optionally 'features_file'.

    Returns:
        tuple: (omics_datasets, omics_networks_tg)
    """
    phenotype_file = dpmon_params['phenotype_file']
    omics_files = dpmon_params['omics_files']
    network_file = dpmon_params['network_file']
    features_file = dpmon_params.get('features_file', None)

    logger.info(f"Phenotype File: {phenotype_file}")
    logger.info(f"omics_files: {omics_files}")
    logger.info(f"network_file: {network_file}")
    logger.info(f"features_file: {features_file}")

    # Load and merge omics and phenotype data
    omics_dataset = get_omics_dataset(phenotype_file, omics_files)
    omics_datasets = get_omics_datasets(omics_dataset, network_file)

    # Load network and create PyTorch Geometric Data objects
    omics_networks_tg = get_omics_networks_tg(network_file,omics_datasets, features_file)

    if not omics_datasets or not omics_networks_tg:
        logger.error("Failed to load omics datasets or networks.")
        raise ValueError("Failed to load omics datasets or networks.")


    return omics_datasets, omics_networks_tg

