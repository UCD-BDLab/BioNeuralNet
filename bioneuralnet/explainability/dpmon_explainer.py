# explainability/dpmon_explainer.py
# Kyle Rohn / Justin Hoang

import torch
from torch import nn
from torch.types import FileLike

from torch_geometric.data import Data
from torch_geometric.explain import Explainer

import pandas as pd

from bioneuralnet.downstream_task import DPMON
from bioneuralnet.downstream_task.dpmon import (
    NeuralNetwork,
    build_omics_networks_tg,
    slice_omics_datasets,
    setup_device
)

from typing import List, Optional


class NeuralNetworkWrapper(nn.Module):
    """A wrapper class for formatting DPMON Neural Network I/O in a form pytorch_geometric requires"""

    def __init__(
        self,
        nn: NeuralNetwork,
        omics_data: pd.DataFrame,
        device
    ):
        """_summary_

        Args:
            nn (NeuralNetwork): _description_
            omics_data (pd.DataFrame): _description_
            device (_type_): _description_
        """

        super(NeuralNetworkWrapper, self).__init__()

        self.nn = nn
        self.training = nn.training
        self.train_features = torch.FloatTensor(omics_data.drop(["phenotype"], axis=1).values).to(device)

    def forward(self, x, edge_index):
        pred, _ = self.nn(self.train_features, x, edge_index)
        return pred


class DPMONExplainer:
    """An explainer for models trained with bioneuralnet.downstream_task.dpmon"""

    def __init__(
        self,
        f: FileLike,
        dpmon: DPMON,
        weights_only: bool = True,
    ):
        """Initialize DPMON explainer object.
        This implementation is a first version.
        There has to be a better way to do this

        Args:
            f (FileLike): The file object or path to a saved model trained with DPMON
            dpmon (DPMON): the DPMON object which trained the model
            weights_only (bool, optional): Load model weights only (you probably want true). Defaults to True.
        """
        # Load weights of model
        model_weights = torch.load(f, weights_only=weights_only)

        # setup device
        device = setup_device(dpmon.gpu, dpmon.cuda)

        # Create Model Data
        combined_omics = pd.concat(dpmon.omics_list, axis=1)
        combined_omics = combined_omics[dpmon.adjacency_matrix.columns]

        combined_omics = combined_omics.merge(
            dpmon.phenotype_data[["phenotype"]],
            left_index=True,
            right_index=True,
        )

        # Maybe save these in dpmon so that replication isn't necessary?
        # As far as I can tell, these return lists for no reason. These lists will always be a single element
        self.omics_data = slice_omics_datasets(
            combined_omics, dpmon.adjacency_matrix
        )[0]
        self.omics_network_tg = build_omics_networks_tg(
            adjacency_matrix=dpmon.adjacency_matrix,
            omics_datasets=[self.omics_data],
            clinical_data=dpmon.clinical_data, # type: ignore
        )[0]

        model = NeuralNetwork(
            model_type=dpmon.model,
            gnn_input_dim=self.omics_network_tg.x.shape[1], # type: ignore
            gnn_hidden_dim=dpmon.gnn_hidden_dim,
            gnn_layer_num=dpmon.layer_num,
            ae_encoding_dim=1,
            nn_input_dim=self.omics_data.drop(["phenotype"], axis=1).shape[1],
            nn_hidden_dim1=dpmon.nn_hidden_dim1,
            nn_hidden_dim2=dpmon.nn_hidden_dim2,
            nn_output_dim=self.omics_data["phenotype"].nunique(),
        )
        model.load_state_dict(model_weights)
        model.eval()
        self.model = NeuralNetworkWrapper(model, self.omics_data, device)


    def explain(self):
        """
        """
        pass
