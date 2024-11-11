# import os
# from args import *
# from dataset import *
# from model import *
# import statistics
# from functools import partial

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix, classification_report

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from torch_geometric.transforms import RandomNodeSplit
# from torchmetrics.classification import MulticlassAUROC

# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# import tempfile
# from ray import train, tune
# from ray.train import Checkpoint


# import logging
# logger = logging.getLogger(__name__)

# # Hyperparameters for the Entire Neural Network Pipeline
# PIPELINE_CONFIGS = {
#     'gnn_layer_num': tune.choice([2, 4, 8, 16, 32, 64, 128]),
#     'gnn_hidden_dim': tune.choice([4, 8, 16, 32, 64, 128]),
#     'lr': tune.loguniform(1e-4, 1e-1),
#     'weight_decay': tune.loguniform(1e-4, 1e-1),
#     'nn_hidden_dim1': tune.choice([4, 8, 16, 32, 64, 128]),
#     'nn_hidden_dim2': tune.choice([4, 8, 16, 32, 64, 128]),
#     'num_epochs': tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),
# }

# HYPERPARAMETERS_TUNING = False

# GNN_MODEL = 'GCN'

# class NeuralNetwork(nn.Module):
#     def __init__(self,
#                  model_type,
#                  gnn_input_dim,
#                  gnn_hidden_dim,
#                  gnn_layer_num,
#                  ae_encoding_dim,
#                  nn_input_dim,
#                  nn_hidden_dim1,
#                  nn_hidden_dim2,
#                  nn_output_dim):
#         super(NeuralNetwork, self).__init__()

#         # Initialize GNN based on model_type
#         if model_type == 'GCN':
#             self.gnn = GCN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
#             logger.info("Initialized GCN layer.")
#         elif model_type == 'GAT':
#             self.gnn = GAT(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
#             logger.info("Initialized GAT layer.")
#         elif model_type == 'SAGE':
#             self.gnn = SAGE(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, output_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
#             logger.info("Initialized SAGE layer.")
#         elif model_type == 'GIN':
#             self.gnn = GIN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, output_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
#             logger.info("Initialized GIN layer.")
#         else:
#             logger.error(f"Unsupported GNN model type: {model_type}")
#             raise ValueError(f"Unsupported GNN model type: {model_type}")

#         # Initialize other components
#         self.autoencoder = Autoencoder(input_dim=gnn_hidden_dim, encoding_dim=ae_encoding_dim)
#         logger.info("Initialized Autoencoder.")
#         self.dim_averaging = DimensionAveraging()
#         logger.info("Initialized DimensionAveraging.")
#         self.predictor = DownstreamTaskNN(nn_input_dim, nn_hidden_dim1, nn_hidden_dim2, nn_output_dim)
#         logger.info("Initialized DownstreamTaskNN.")


#         # self.gnn = ''  # GNN Model
#         # if GNN_MODEL == 'GCN':
#         #     self.gnn = GCN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)

#         # # TODO: Revisit the Autoencoder - Since we Only Need to Encode so the Model doesn't learn here!
#         # self.autoencoder = Autoencoder(input_dim=gnn_hidden_dim, encoding_dim=ae_encoding_dim)
#         # self.dim_averaging = DimensionAveraging()
#         # self.predictor = DownstreamTaskNN(nn_input_dim, nn_hidden_dim1, nn_hidden_dim2, nn_output_dim)
#         # # self.predictor = LogisticRegression(nn_input_dim, nn_output_dim)



#     def forward(self, omics_dataset, omics_network_tg):
#         # print(omics_network_tg)
#         omics_network_nodes_embedding = self.gnn(omics_network_tg)

#         # print("Dimenions of Embeddings after GCN")
#         # print(omics_network_nodes_embedding.shape)
#         # print(omics_network_nodes_embedding)
#         omics_network_nodes_embedding_ae = self.autoencoder(omics_network_nodes_embedding)
#         # print("Scaling Factors %s" % omics_network_nodes_embedding_ae)
#         # print("Embeddings %s" % omics_network_nodes_embedding_ae)

#         # print("Dimensions of Embeddings after Autoencoder")
#         # print(omics_network_nodes_embedding_ae.shape)
#         # print(omics_network_nodes_embedding_ae)

#         omics_network_nodes_embedding_avg = self.dim_averaging(omics_network_nodes_embedding_ae)

#         # print("Dimensions of Embeddings after Averaging")
#         # print(omics_network_nodes_embedding_avg.shape)
#         # print("Averaging Embeddings %s" % omics_network_nodes_embedding_avg)

#         # Multiplying the original_dataset with the generated embeddings
#         omics_dataset_with_embeddings = torch.mul(omics_dataset,
#                                                   omics_network_nodes_embedding_avg.expand(omics_dataset.shape[1],
#                                                                                            omics_dataset.shape[0]).t())
#         predictions = self.predictor(omics_dataset_with_embeddings)

#         return predictions, omics_dataset_with_embeddings


# def train_n(config, omics_dataset, omics_network_tg):
#     X = omics_dataset.drop(['finalgold_visit'], axis=1)
#     Y = omics_dataset['finalgold_visit']
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#     # TODO: Double Check if the Dataset is already Standardized
#     # scaler = StandardScaler()
#     # X_train_scaled = scaler.fit_transform(X_train.to_numpy())
#     # X_test_scaled = scaler.transform(X_test.to_numpy())

#     # Convert to PyTorch Tensors
#     X_train_tensor = torch.FloatTensor(X_train.to_numpy())
#     y_train_tensor = torch.LongTensor(y_train.to_numpy())

#     X_test_tensor = torch.FloatTensor(X_test.to_numpy())
#     y_test_tensor = torch.LongTensor(y_test.to_numpy())

#     gnn_input_dim = omics_network_tg.x.shape[1]
#     nn_input_dim = X_train_tensor.shape[1]
#     nn_output_dim = 6  # Number of Classes # TODO: If we Merge Classes; then Update the Number of Classes here

#     # lr_model = LogisticRegression(nn_input_dim, nn_output_dim)
#     nn_model = NeuralNetwork(gnn_input_dim=gnn_input_dim,
#                              gnn_hidden_dim=config['gnn_hidden_dim'],
#                               gnn_layer_num=config['gnn_layer_num'],
#                               ae_encoding_dim=1,
#                               nn_input_dim=nn_input_dim, nn_hidden_dim1=config['nn_hidden_dim1'],
#                               nn_hidden_dim2=config['nn_hidden_dim2'],
#                               nn_output_dim=nn_output_dim)

#     # Define the Loss Function and the Optimizer for the NN Model
#     nn_criterion = nn.CrossEntropyLoss()
#     nn_optimizer = optim.Adam(nn_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

#     # Define the Loss Function and the Optimizer for the LR Model
#     # lr_criterion = nn.CrossEntropyLoss()
#     # lr_optimizer = optim.Adam(lr_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

#     # Training loop
#     for epoch in range(config['num_epochs']):
#         nn_optimizer.zero_grad()
#         # lr_optimizer.zero_grad()
#         # Forward Pass
#         output, omics_dataset_with_embeddings = nn_model(X_train_tensor, omics_network_tg)
#         # lr_output = lr_model(X_train_tensor)
#         # Compute the Loss Value
#         loss = nn_criterion(output, y_train_tensor)
#         # lr_loss = lr_criterion(lr_output, y_train_tensor)

#         # Backward Propagation
#         loss.backward()
#         nn_optimizer.step()

#         # lr_loss.backward()
#         # lr_optimizer.step()

#         # Print the Loss Value for Monitoring
#         # print(f"Epoch [{epoch + 1}/{epoch}], Loss: {loss.item()}")

#         if HYPERPARAMETERS_TUNING:
#             metrics = {"loss": loss.item()}
#             with tempfile.TemporaryDirectory() as tempdir:
#                 torch.save(
#                     {"epoch": epoch, "model_state": nn_model.state_dict()},
#                     os.path.join(tempdir, "checkpoint.pt"),
#                 )
#                 train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

#         with torch.no_grad():
#             nn_model.eval()
#             # lr_model.eval()

#             # lr_preds = lr_model(X_test_tensor)
#             # l, predicted = torch.max(lr_preds, 1)
#             # accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)
#             # print("Accuracy with LR: {:.4f}".format(accuracy))

#             preds, omics_dataset_with_embeddings = nn_model(X_test_tensor, omics_network_tg)
#             l, predicted = torch.max(preds, 1)
#             accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)

#             # confusion_mtrx = confusion_matrix(y_test_tensor, predicted)
#             # mc_auroc = MulticlassAUROC(num_classes=3, average='macro', thresholds=None)
#             # print(f'Test Accuracy: {accuracy}')
#             # print(f'Classification Report: {classification_report(y_test_tensor, predicted)}')
#             # print(f'ROC AUC Score: {mc_auroc(outputs, y_test_tensor)}')
#             # print(f'Confusion Matrix: {confusion_mtrx}')

#             if HYPERPARAMETERS_TUNING:
#                 metrics["accuracy"] = accuracy
#                 with tempfile.TemporaryDirectory() as tempdir:
#                     train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

#     return accuracy


# def main():
#     args = make_args()

#     # Setting up GPU Vs. CPU Usage
#     if args.gpu:
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
#         print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
#     else:
#         print('Using CPU')

#     # Setting up Hyperparameter Tuning
#     if args.tune:
#         print("Tuning the Hyperparameters of the Pipeline")
#         global HYPERPARAMETERS_TUNING
#         HYPERPARAMETERS_TUNING = True


#     accuracies = []
#     omics_datasets, omics_networks_tg = get_dataset()
#     # print(omics_datasets)
#     # print(omics_networks_tg)

#     if args.tune:
#         os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
#         for (omics_dataset, omics_network_tg) in zip(omics_datasets, omics_networks_tg):
#             reporter = CLIReporter(metric_columns=["loss"])
#             # Scheduler to Stop Bad Performing Trials
#             scheduler = ASHAScheduler(
#                 metric="loss",  # TODO: Consider using Accuracy as a Metric
#                 mode="min",
#                 grace_period=10,
#                 reduction_factor=2)

#             NUM_SAMPLES = 10  # TODO: Need to Run without Limiting the Number of Samples
#             result = tune.run(partial(train_n, omics_dataset=omics_dataset, omics_network_tg=omics_network_tg),
#                               resources_per_trial={"cpu": 8, "gpu": 0},
#                               config=PIPELINE_CONFIGS, num_samples=NUM_SAMPLES, scheduler=scheduler,
#                               name='TuningCurrentSmokers',  # TODO: Update based on the Name of the Network
#                               keep_checkpoints_num=1,
#                               checkpoint_score_attr='min-loss',
#                               progress_reporter=reporter
#                               )
#             best_trial = result.get_best_trial("loss", "min", "last")
#             print("Best trial config: {}".format(best_trial.config))
#             print("Best trial final test loss: {}".format(best_trial.last_result["loss"]))
#     else:
#         for (omics_dataset, omics_network_tg) in zip(omics_datasets, omics_networks_tg):
#             for i in range(args.repeat_num):
#                 accuracy = train_n(config={
#                     'gnn_layer_num': 4,
#                     'gnn_hidden_dim': 8,
#                     'lr': 0.029352058542109778,
#                     'weight_decay': 0.0010178240820048331,
#                     'nn_hidden_dim1': 8,
#                     'nn_hidden_dim2': 32,
#                     'num_epochs': 100,
#                 }, omics_dataset=omics_dataset, omics_network_tg=omics_network_tg)
#                 print("Accuracy: {}".format(accuracy))
#                 accuracies.append(accuracy)
#             print("Best Accuracy: {}".format(max(accuracies)))
#             print("Average Accuracy is {}".format(sum(accuracies) / len(accuracies)))
#             print("Standard Deviation for Avg Accuracy is {}".format(statistics.stdev(accuracies)))

import os
import logging
import torch
import pandas as pd
import torch.nn as nn
from dataset import get_dataset
from model import NeuralNetwork
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import tempfile
import statistics

# Initialize the module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Ensure that INFO level logs are displayed

def train_model(model, criterion, optimizer, train_data, train_labels, epoch_num, tune_flag=False):
    """
    Train the given model on the provided training data.

    Args:
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_data (torch.Tensor): The training features.
        train_labels (dict): Dictionary containing training labels and network.
        epoch_num (int): Number of training epochs.
        tune_flag (bool): Flag to indicate if hyperparameter tuning is active.

    Returns:
        float: The final training accuracy.
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

        # Hyperparameter tuning reporting
        if tune_flag:
            tune.report(loss=loss.item())

    # Evaluation after training
    model.eval()
    with torch.no_grad():
        predictions, _ = model(train_data, train_labels['omics_network'])
        _, predicted = torch.max(predictions, 1)
        accuracy = (predicted == train_labels['labels']).sum().item() / len(train_labels['labels'])
        logger.info(f"Training Accuracy: {accuracy:.4f}")

        if tune_flag:
            tune.report(accuracy=accuracy)

    return accuracy

def run_dpmon(dpmon_params, output_dir):
    """
    Run the DPMON prediction model with the given parameters.

    Args:
        dpmon_params (dict): Dictionary containing model and training parameters.
        output_dir (str): Directory where outputs will be saved.
    """
    # Extract parameters
    model_type = dpmon_params['model']
    gpu = dpmon_params['gpu']
    cuda = dpmon_params['cuda']
    tune_flag = dpmon_params['tune']

    # Handle GPU configuration
    if gpu:
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
    if not omics_datasets or not omics_networks_tg:
        logger.error("No datasets or networks found.")
        raise ValueError("No datasets or networks found.")

    if tune_flag:
        run_hyperparameter_tuning(omics_datasets, omics_networks_tg, model_type, device)

    else:
        run_standard_training(dpmon_params,device, omics_datasets, omics_networks_tg, output_dir)

    logger.info("DPMON run completed successfully.")


def run_standard_training(dpmon_params, device, omics_datasets, omics_networks_tg, output_dir):

    model_type = dpmon_params['model']
    lr = dpmon_params['lr']
    weight_decay = dpmon_params['weight_decay']
    layer_num = dpmon_params['layer_num']
    hidden_dim = dpmon_params['hidden_dim']
    epoch_num = dpmon_params['epoch_num']
    repeat_num = dpmon_params['repeat_num']
    nn_hidden_dim1 = dpmon_params['nn_hidden_dim1']
    nn_hidden_dim2 = dpmon_params['nn_hidden_dim2']

    # Standard training without hyperparameter tuning
    for omics_dataset, omics_network in zip(omics_datasets, omics_networks_tg):
        logger.info(f"Dataset shape: {omics_dataset.shape}")
        logger.info(f"Network nodes: {omics_network.x.shape[0]}, Features: {omics_network.x.shape[1]}")
        accuracies = []
        for i in range(repeat_num):
            logger.info(f"Starting training iteration {i+1}/{repeat_num}")

            # Initialize model with dynamic model_type
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

            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Prepare training data
            train_features = torch.FloatTensor(omics_dataset.drop(['finalgold_visit'], axis=1).values).to(device)
            train_labels = {
                'labels': torch.LongTensor(omics_dataset['finalgold_visit'].values).to(device),
                'omics_network': omics_network.to(device)
            }

            # Train the model using the separated training function
            accuracy = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train_data=train_features,
                train_labels=train_labels,
                epoch_num=epoch_num,
                tune_flag=False
            )
            accuracies.append(accuracy)

            # Save the model
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

        # After all repeats, log statistics
        if accuracies:
            max_accuracy = max(accuracies)
            avg_accuracy = sum(accuracies) / len(accuracies)
            std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            logger.info(f"Best Accuracy: {max_accuracy:.4f}")
            logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
            logger.info(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")


def run_hyperparameter_tuning(dpmon_params,omics_datasets, omics_networks_tg, device):
    """
    Run hyperparameter tuning for the DPMON model.
    """
    # Define hyperparameter search space (similar to Code 1's PIPELINE_CONFIGS)
    pipeline_configs = {
        'gnn_layer_num': tune.choice([2, 4, 8, 16, 32, 64, 128]),
        'gnn_hidden_dim': tune.choice([4, 8, 16, 32, 64, 128]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'weight_decay': tune.loguniform(1e-4, 1e-1),
        'nn_hidden_dim1': tune.choice([4, 8, 16, 32, 64, 128]),
        'nn_hidden_dim2': tune.choice([4, 8, 16, 32, 64, 128]),
        'num_epochs': tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),
    }
    pass_device = device
    # Define the training function for Ray Tune
    def tune_train_n(config, omics_dataset, omics_network_tg, pass_device):
        device = pass_device
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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        train_features = torch.FloatTensor(omics_dataset.drop(['finalgold_visit'], axis=1).values).to(device)
        train_labels = {
            'labels': torch.LongTensor(omics_dataset['finalgold_visit'].values).to(device),
            'omics_network': omics_network_tg.to(device)
        }

        accuracy = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_data=train_features,
            train_labels=train_labels,
            epoch_num=config['num_epochs'],
            tune_flag=True
        )

        return accuracy

    # Setup Ray Tune scheduler and reporter
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        grace_period=10,
        reduction_factor=2
    )

    # Iterate over datasets and networks
    gpu = dpmon_params['gpu']
    for (omics_dataset, omics_network_tg) in zip(omics_datasets, omics_networks_tg):
        result = tune.run(
            tune.with_parameters(tune_train_n, omics_dataset=omics_dataset, omics_network_tg=omics_network_tg),
            resources_per_trial={"cpu": 8, "gpu": 1 if gpu else 0},
            config=pipeline_configs,
            num_samples=10,
            scheduler=scheduler,
            name='TuningCurrentSmokers',
            progress_reporter=reporter,
            keep_checkpoints_num=1,
            checkpoint_score_attr='min-loss'
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        logger.info("Best trial config: {}".format(best_trial.config))
        logger.info("Best trial final test loss: {}".format(best_trial.last_result["loss"]))
        logger.info("Best trial final test accuracy: {}".format(best_trial.last_result["accuracy"]))