import os
import logging
import torch
import pandas as pd
import torch.nn as nn
from .dataset import get_dataset
from .model import NeuralNetwork
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import statistics
from functools import partial
from ray import train
from ray.train import Checkpoint
import tempfile

# setting up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(model, criterion, optimizer, train_data, train_labels, epoch_num):
    """
    Train the neural network model on the provided training data.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        train_data (torch.Tensor): The input features for training.
        train_labels (dict): A dictionary containing:
            - 'labels' (torch.Tensor): The ground truth labels.
            - 'omics_network' (torch_geometric.data.Data): The omics network data.
        epoch_num (int): The number of epochs to train the model.

    Returns:
        float: The final training accuracy after all epochs.
    """
    model.train()
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        outputs, _ = model(train_data, train_labels['omics_network'])
        loss = criterion(outputs, train_labels['labels'])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epoch_num}], Loss: {loss.item():.4f}")

    # Evaluation after training
    model.eval()
    with torch.no_grad():
        predictions, _ = model(train_data, train_labels['omics_network'])
        _, predicted = torch.max(predictions, 1)
        accuracy = (predicted == train_labels['labels']).sum().item() / len(train_labels['labels'])
        logger.info(f"Training Accuracy: {accuracy:.4f}")

    return accuracy

def run_dpmon(dpmon_params, output_dir):
    """
    Execute the DPMON prediction model with the provided parameters.

    Args:
        dpmon_params (dict): Dictionary containing model and training parameters.
        output_dir (str): Directory where model outputs and artifacts will be saved.

    Raises:
        ValueError: If no datasets or networks are found.
    """
    # Extract parameters
    model_type = dpmon_params['model']
    gpu = dpmon_params['gpu']
    cuda = dpmon_params['cuda']
    tune_flag = dpmon_params['tune']

    # Handle GPU configuration
    if gpu:
        logger.info("GPU Flag set to True: Using GPU for training.")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f'Using GPU {cuda}')
        else:
            logger.warning(f'GPU {cuda} requested but not available, using CPU')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')

    # Load datasets
    try:
        omics_datasets, omics_networks_tg = get_dataset(dpmon_params)
        logger.info(f"Loaded dataset with {len(omics_datasets)} datasets and {len(omics_networks_tg)} networks.")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

    # Check if datasets are available
    logger.debug(f"Datasets: {len(omics_datasets)}, Networks: {len(omics_networks_tg)}")
    if not omics_datasets or not omics_networks_tg:
        logger.error("No datasets or networks found.")
        raise ValueError("No datasets or networks found.")

    if tune_flag:
        # Run hyperparameter tuning
        logger.info("Tune Flag set to True: Running hyperparameter tuning.")
        run_hyperparameter_tuning(dpmon_params, omics_datasets, omics_networks_tg, device)
    else:
        # Run standard training
        logger.info("Tune Flag set to False: Running standard training.")
        run_standard_training(dpmon_params, device, omics_datasets, omics_networks_tg, output_dir)

    logger.info("DPMON run completed successfully.")

def run_standard_training(dpmon_params, device, omics_datasets, omics_networks_tg, output_dir):
    """
    Perform standard training without hyperparameter tuning.

    Args:
        dpmon_params (dict): Dictionary containing model and training parameters.
        device (torch.device): The device to run the model on (CPU or GPU).
        omics_datasets (list): List of omics datasets.
        omics_networks_tg (list): List of omics networks in PyTorch Geometric format.
        output_dir (str): Directory where model outputs and artifacts will be saved.
    """
    # Extract training parameters
    model_type = dpmon_params['model']
    layer_num = dpmon_params['layer_num']
    hidden_dim = dpmon_params['gnn_hidden_dim']
    epoch_num = dpmon_params['epoch_num']
    repeat_num = dpmon_params['repeat_num']
    nn_hidden_dim1 = dpmon_params['nn_hidden_dim1']
    nn_hidden_dim2 = dpmon_params['nn_hidden_dim2']

    # Loop over each dataset and network
    for omics_dataset, omics_network in zip(omics_datasets, omics_networks_tg):
        logger.info(f"Dataset shape: {omics_dataset.shape}")
        logger.info(f"Network nodes: {omics_network.x.shape[0]}, Features: {omics_network.x.shape[1]}")
        accuracies = []
        for i in range(repeat_num):
            logger.info(f"Starting training iteration {i+1}/{repeat_num}")

            # Initialize the neural network model
            try:
                model = NeuralNetwork(
                    model_type=model_type,
                    gnn_input_dim=omics_network.x.shape[1],
                    gnn_hidden_dim=hidden_dim,
                    gnn_layer_num=layer_num,
                    ae_encoding_dim=1,
                    nn_input_dim=omics_dataset.drop(['finalgold_visit'], axis=1).shape[1],
                    nn_hidden_dim1=nn_hidden_dim1,
                    nn_hidden_dim2=nn_hidden_dim2,
                    nn_output_dim=omics_dataset['finalgold_visit'].nunique()
                ).to(device)
                logger.info("Model initialized successfully.")
            except ValueError as ve:
                logger.error(f"Model initialization failed: {ve}")
                raise

            # Define loss function and optimizer
            logger.info("Defining loss and optimizer.")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=dpmon_params['lr'], weight_decay=dpmon_params['weight_decay'])
            
            # Prepare training data
            logger.info("Preparing training data.")
            train_features = torch.FloatTensor(omics_dataset.drop(['finalgold_visit'], axis=1).values).to(device)
            train_labels = {
                'labels': torch.LongTensor(omics_dataset['finalgold_visit'].values.copy()).to(device),
                'omics_network': omics_network.to(device)
            }

            # Train the model
            logger.info("Training the model.")
            accuracy = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_data=train_features,
                train_labels=train_labels,
                epoch_num=epoch_num
            )
            accuracies.append(accuracy)

            # Save the trained model
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f'dpm_model_iter_{i+1}.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

            # Save predictions
            model.eval()
            with torch.no_grad():
                predictions, _ = model(train_features, omics_network.to(device))
                _, predicted = torch.max(predictions, 1)
                predictions_path = os.path.join(output_dir, f'predictions_iter_{i+1}.csv')
                predictions_df = pd.DataFrame({
                    'Actual': omics_dataset['finalgold_visit'],
                    'Predicted': predicted.cpu().numpy()
                })
                predictions_df.to_csv(predictions_path, index=False)
                logger.info(f"Predictions saved to {predictions_path}")

        # Logging statistics after all repeats
        logger.info(f"Training completed for dataset with shape {omics_dataset.shape} and network with {omics_network.x.shape[0]} nodes.")
        if accuracies:
            max_accuracy = max(accuracies)
            avg_accuracy = sum(accuracies) / len(accuracies)
            std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            logger.info(f"Best Accuracy: {max_accuracy:.4f}")
            logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
            logger.info(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

def run_hyperparameter_tuning(dpmon_params, omics_datasets, omics_networks_tg, device):
    """
    Perform hyperparameter tuning using Ray Tune.

    Args:
        dpmon_params (dict): Dictionary containing model and training parameters.
        omics_datasets (list): List of omics datasets.
        omics_networks_tg (list): List of omics networks in PyTorch Geometric format.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    # Define the hyperparameter search space
    pipeline_configs = {
        # 'gnn_layer_num': tune.choice([2, 4, 8, 16, 32, 64, 128]),
        # 'gnn_hidden_dim': tune.choice([4, 8, 16, 32, 64, 128]),
        # 'lr': tune.loguniform(1e-4, 1e-1),
        # 'weight_decay': tune.loguniform(1e-4, 1e-1),
        # 'nn_hidden_dim1': tune.choice([4, 8, 16, 32, 64, 128]),
        # 'nn_hidden_dim2': tune.choice([4, 8, 16, 32, 64, 128]),
        # 'num_epochs': tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),

        # Reduced search space for faster tuning (Testing Olnly)
        'gnn_layer_num': tune.choice([2, 4, 8]),
        'gnn_hidden_dim': tune.choice([4, 8, 16]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'weight_decay': tune.loguniform(1e-4, 1e-1),
        'nn_hidden_dim1': tune.choice([4, 8, 16]),
        'nn_hidden_dim2': tune.choice([4, 8, 16]),
        'num_epochs': tune.choice([2, 16, 64]),
    }

    def tune_train_n(config, omics_dataset, omics_network_tg):
        """
        Training function used by Ray Tune for hyperparameter tuning.

        Args:
            config (dict): Configuration dictionary with hyperparameters.
            omics_dataset (pd.DataFrame): The omics dataset for training.
            omics_network_tg (torch_geometric.data.Data): The omics network data.
        """
        # Initialize the neural network model with hyperparameters from config
        model = NeuralNetwork(
            model_type=dpmon_params['model'],
            gnn_input_dim=omics_network_tg.x.shape[1],
            gnn_hidden_dim=config['gnn_hidden_dim'],
            gnn_layer_num=config['gnn_layer_num'],
            ae_encoding_dim=1,
            nn_input_dim=omics_dataset.drop(['finalgold_visit'], axis=1).shape[1],
            nn_hidden_dim1=config['nn_hidden_dim1'],
            nn_hidden_dim2=config['nn_hidden_dim2'],
            nn_output_dim=omics_dataset['finalgold_visit'].nunique()
        ).to(device)

        # Define loss function and optimizer
        logger.info("Defining loss and optimizer.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        # Prepare training data
        logger.info("Preparing training data.")
        train_features = torch.FloatTensor(omics_dataset.drop(['finalgold_visit'], axis=1).values).to(device)
        train_labels = {
            'labels': torch.LongTensor(omics_dataset['finalgold_visit'].values.copy()).to(device),
            'omics_network': omics_network_tg.to(device)
        }

        # Training loop
        logger.info("Starting training loop.")
        for epoch in range(config['num_epochs']):
            model.train()
            optimizer.zero_grad()
            outputs, _ = model(train_features, train_labels['omics_network'])
            loss = criterion(outputs, train_labels['labels'])
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            #logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {loss.item():.4f}")
            _, predicted = torch.max(outputs, 1)
            total = train_labels['labels'].size(0)
            correct = (predicted == train_labels['labels']).sum().item()
            accuracy = correct / total

            # Reporing metrics to Ray Tune
            metrics = {"loss": loss.item(), "accuracy": accuracy}
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

        # Final evaluation after training
        logger.info("Final evaluation after training.")
        model.eval()
        with torch.no_grad():
            outputs, _ = model(train_features, train_labels['omics_network'])
            loss = criterion(outputs, train_labels['labels'])
            _, predicted = torch.max(outputs, 1)
            total = train_labels['labels'].size(0)
            correct = (predicted == train_labels['labels']).sum().item()
            accuracy = correct / total

            # Report final metrics
            metrics = {"loss": loss.item(), "accuracy": accuracy}
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": config['num_epochs'], "model_state": model.state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

    # Set up Ray Tune scheduler and reporter
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        grace_period=10,
        reduction_factor=2
    )

    # Iterate over datasets and networks
    logger.info("Starting hyperparameter tuning.")
    gpu = dpmon_params['gpu']
    for (omics_dataset, omics_network_tg) in zip(omics_datasets, omics_networks_tg):
        # Disable strict metric checking
        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
        logger.info(f"Starting hyperparameter tuning for dataset with shape {omics_dataset.shape} and network with {omics_network_tg.x.shape[0]} nodes.")

        # Start hyperparameter tuning with Ray Tune
        result = tune.run(
            partial(tune_train_n, omics_dataset=omics_dataset, omics_network_tg=omics_network_tg),
            resources_per_trial={"cpu": 2, "gpu": 1 if gpu else 0},
            config=pipeline_configs,
            num_samples=10,
            scheduler=scheduler,
            name='Hyperparameter_Tuning',
            progress_reporter=reporter,
            keep_checkpoints_num=1,
            checkpoint_score_attr='min-loss'
        )

        # Retrieve the best trial
        best_trial = result.get_best_trial("loss", "min", "last")
        logger.info("Best trial config: {}".format(best_trial.config))
        logger.info("Best trial final loss: {}".format(best_trial.last_result["loss"]))
        logger.info("Best trial final accuracy: {}".format(best_trial.last_result["accuracy"]))
