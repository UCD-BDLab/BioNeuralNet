# explainability/dpmon_explainer.py
# Kyle Rohn / Justin Hoang

import torch
import os
from torch import nn
from torch.types import FileLike

from typing import Literal

from torch_geometric.data import Data
from torch_geometric.explain import Explainer, ExplainerAlgorithm, ModelConfig

import pandas as pd

from bioneuralnet.downstream_task import DPMON
from bioneuralnet.downstream_task.dpmon import (
    NeuralNetwork,
    prepare_node_features,
    slice_omics_datasets,
    setup_device,
)

from typing import List, Optional


class NeuralNetworkWrapper(nn.Module):
    """A wrapper class for formatting DPMON Neural Network IO in a form pytorch_geometric requires"""

    def __init__(self, nn: NeuralNetwork):
        """_summary_

        Args:
            nn (NeuralNetwork): _description_
            omics_data (pd.DataFrame): _description_
            device (_type_): _description_
        """

        super(NeuralNetworkWrapper, self).__init__()

        self.nn = nn

    def forward(self, x, edge_index, train_features, edge_attr=None):

        _omics_network_tg = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        pred, _, _ = self.nn(train_features, _omics_network_tg)
        return pred


class DPMONExplainer:
    """An explainer for models trained with bioneuralnet.downstream_task.dpmon"""

    def __init__(
        self,
        f: FileLike,
        dpmon: DPMON,
        algorithm: ExplainerAlgorithm,
        mode: Literal["regression", "binary_classification", "multiclass_classification"],
        explanation_type: Literal["model", "phenomenon"] = "model",
        node_mask_type: Literal["object", "common_attributes", "attributes"] | None = "attributes",
        edge_mask_type: Literal["object", "common_attributes", "attributes"] | None = "object",
        task_level: Literal["edge", "node", "graph"] = "graph",
        return_type: Literal["raw", "log_probs", "probs"] = "raw",
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

        self.omics_dataset = slice_omics_datasets(
            dpmon.combined_omics, dpmon.adjacency_matrix, dpmon.phenotype_col
        )[0]

        self.omics_network_tg = prepare_node_features(
            adjacency_matrix=dpmon.adjacency_matrix,
            omics_datasets=[self.omics_dataset],
            clinical_data=dpmon.clinical_data,
            phenotype_col="phenotype",
        )[0]

        model = NeuralNetwork(
            model_type=dpmon.model,
            gnn_input_dim=self.omics_network_tg.x.shape[1],  # type: ignore
            gnn_hidden_dim=dpmon.gnn_hidden_dim,
            gnn_layer_num=dpmon.gnn_layer_num,
            dim_reduction=dpmon.dim_reduction,
            ae_encoding_dim=dpmon.ae_encoding_dim,
            nn_input_dim=self.omics_dataset.drop(["phenotype"], axis=1).shape[1],
            nn_hidden_dim1=dpmon.nn_hidden_dim1,
            nn_hidden_dim2=dpmon.nn_hidden_dim2,
            nn_output_dim=self.omics_dataset["phenotype"].nunique(),
            gnn_activation=dpmon.gnn_activation,
        )

        model.load_state_dict(model_weights)
        model.eval()
        
        self.train_features = torch.FloatTensor(
            self.omics_dataset.drop(["phenotype"], axis=1).values
        ).to(device)

        self.model = NeuralNetworkWrapper(model)
        self.explainer = Explainer(
            self.model,
            algorithm,
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=ModelConfig(
                mode=mode,
                task_level=task_level,
                return_type=return_type
            )
        )
        
        if self.omics_network_tg.x != None and self.omics_network_tg.edge_index != None:
            self.expl = self.explainer(**self.omics_network_tg.to_dict(), train_features=self.train_features)
            print(self.expl.edge_mask)
            print(self.expl.node_mask)
        
        
        
            
    def visualize_feature_importance(self, path: os.PathLike):
        self.expl.visualize_feature_importance(str(path))
    
    def visualize_graph(self, path: os.PathLike):
        self.expl.visualize_graph(str(path))
