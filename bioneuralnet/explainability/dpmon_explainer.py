# explainability/dpmon_explainer.py
# Kyle Rohn / Justin Hoang

import torch
from torch.types import FileLike

from torch_geometric.explain import Explainer

import pandas as pd

from bioneuralnet.downstream_task import DPMON
from bioneuralnet.downstream_task.dpmon import (
    NeuralNetwork,
    build_omics_networks_tg,
    slice_omics_datasets,
)

from typing import List, Optional


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

        # Create Model Data
        combined_omics = pd.concat(dpmon.omics_list, axis=1)
        combined_omics = combined_omics[dpmon.adjacency_matrix.columns]

        combined_omics = combined_omics.merge(
            dpmon.phenotype_data[["phenotype"]],
            left_index=True,
            right_index=True,
        )

        # Maybe save these in dpmon so that replication isn't necessary maybe?
        self.omics_dataset = slice_omics_datasets(
            combined_omics, dpmon.adjacency_matrix
        )
        self.omics_networks_tg = build_omics_networks_tg(
            adjacency_matrix=dpmon.adjacency_matrix,
            omics_datasets=self.omics_dataset,
            clinical_data=dpmon.clinical_data,
        )

        # Initialize model(s)? to insert weights into based on dpmon params
        # unsure why there are multiple layers of data
        self.models = []
        for omics_data, omics_network_tg in zip(
            self.omics_dataset, self.omics_networks_tg
        ):
            model = NeuralNetwork(
                    model_type=dpmon.model,
                    gnn_input_dim=omics_network_tg.x.shape[1],
                    gnn_hidden_dim=dpmon.gnn_hidden_dim,
                    gnn_layer_num=dpmon.layer_num,
                    ae_encoding_dim=1,
                    nn_input_dim=omics_data.drop(["phenotype"], axis=1).shape[1],
                    nn_hidden_dim1=dpmon.nn_hidden_dim1,
                    nn_hidden_dim2=dpmon.nn_hidden_dim2,
                    nn_output_dim=omics_data["phenotype"].nunique(),
                )
            model.load_state_dict(model_weights)
            # this will cause issues if the model trained on the data isn't the same one
            # either needs a way to enumerate data/models or dynamically determine if it is the correct one
            self.models.append(model)

    def explain(self):
        """
        We don't have all of the details ironed out for this
        What we do know:
        - There are multiple types of layers in this GNN
            - gnn layer
            - autoencoder layer
            - downstream task layer

        The reality is the pytorch_geometric explainer doesn't interface well with this model
        This leaves us a couple options:
        - Generate explanations at only the GNN level
        - Create custom explainer which interfaces with pytorch_geometric
        - Change design of NeuralNetwork to interface with pytorch_geometric
        """
        pass
