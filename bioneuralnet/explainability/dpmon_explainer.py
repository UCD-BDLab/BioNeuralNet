# explainability/dpmon_explainer.py
# Kyle Rohn / Justin Hoang

import torch
import os
from torch import nn
from torch.types import FileLike

from typing import Literal, Dict, List

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

    def forward(self, x, edge_index, train_features, **kwargs):

        _omics_network_tg = Data(x=x, edge_index=edge_index, **kwargs)

        pred, _, _ = self.nn(train_features, _omics_network_tg)
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
        This implementation is a first version. By default, it uses `torch_geometric.explain.GNNExplainer()`
        to produce feature importance explanations on `clinical` data. The raw node explanations are stored in
        `self.expl.node_mask`

        It is important to note that these explanations likely do not capture the full picture of the predictions
        in a multi-omics network, rather provide insight into the clinical (patient) features which account
        for the explanations.

        This explainer object also produces edge importances at an `object` level. It is currently
        unknown how useful these explanations are, but are stored in `self.expl.edge_mask`. A method is provided
        to retrieve the top n important edges for the user to observe if they want.

        Otherwise, all of the default `pytorch_geometric.explain.Explainer` methods are available to use, whether
        through a provided wrapper or through the `expl` member variable

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

        self.clinical_data = dpmon.clinical_data

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

    def explain(
        self,
        algorithm: ExplainerAlgorithm,
        mode: Literal[
            "regression", "binary_classification", "multiclass_classification"
        ],
        explanation_type: Literal["model", "phenomenon"] = "model",
        node_mask_type: (
            Literal["object", "common_attributes", "attributes"] | None
        ) = "attributes",
        edge_mask_type: (
            Literal["object", "common_attributes", "attributes"] | None
        ) = "object",
        task_level: Literal["edge", "node", "graph"] = "graph",
        return_type: Literal["raw", "log_probs", "probs"] = "raw",
    ):
        """Generate explanations for the DPMON instance and the model loaded at the specified path

        Args:
            algorithm (ExplainerAlgorithm): The `pytorch_geometric.explain` explainer algorithm to use. Currently only tested with `GNNExplainer()`
            mode (Literal[ &quot;regression&quot;, &quot;binary_classification&quot;, &quot;multiclass_classification&quot; ]): The type of prediction the GNN is making
            explanation_type (Literal[&quot;model&quot;, &quot;phenomenon&quot;], optional): Whether to generate explanations on the `model` predictions or Explains the `phenomenon` that the model is trying to predict. Defaults to "model".
            node_mask_type (Literal[&quot;object&quot;, &quot;common_attributes&quot;, &quot;attributes&quot;]  |  None, optional): The node explanation type to generate. Defaults to "attributes".
            edge_mask_type (Literal[&quot;object&quot;, &quot;common_attributes&quot;, &quot;attributes&quot;]  |  None, optional): The edge explanation type to generate. Defaults to "object".
            task_level (Literal[&quot;edge&quot;, &quot;node&quot;, &quot;graph&quot;], optional): The prediction scope of the model. Defaults to "graph".
            return_type (Literal[&quot;raw&quot;, &quot;log_probs&quot;, &quot;probs&quot;], optional): The output of the model. Defaults to "raw".
        """
        self.explainer = Explainer(
            self.model,
            algorithm,
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=ModelConfig(
                mode=mode, task_level=task_level, return_type=return_type
            ),
        )

        if self.omics_network_tg.x != None and self.omics_network_tg.edge_index != None:
            self.expl = self.explainer(
                **self.omics_network_tg.to_dict(), train_features=self.train_features
            )

    def edge_importance_data(self, top_n: int = 5) -> List[Dict]:
        """Method for providing a summary on object level edge importance

        Args:
            top_n (int, optional): the number of important edges to retrieve. Defaults to 5.
        Returns:
            List[Dict]: The edges with the top n importances and `edge_attr` if it exists
        """

        edges = []

        if self.expl.edge_mask != None:
            for idx, importance in enumerate(self.expl.edge_mask):
                importance = importance.item()
                new_edge = {"importance": importance, "edge": self.omics_network_tg.edge_index[:, idx]}  # type: ignore
                if self.omics_network_tg.edge_attr != None:
                    if self.omics_network_tg.edge_attr.ndim == 1:
                        new_edge.update(
                            {"edge_attr": self.omics_network_tg.edge_attr[idx].item()}
                        )
                    else:
                        new_edge.update(
                            {"edge_attr": self.omics_network_tg.edge_attr[idx, :]}
                        )

                if len(edges) < 1:
                    edges.append(new_edge)  # type: ignore
                    continue

                for i, edge in enumerate(edges):
                    if edge["importance"] < importance:
                        edges.insert(i, new_edge)  # type: ignore
                        break
                edges = edges[:top_n]
            return edges[:top_n]
        else:
            raise AttributeError(
                "edge_mask is not defined. Generate explanations on edges first"
            )

    def visualize_feature_importance(
        self, path: os.PathLike | None = None, top_k: int | None = None
    ):
        """Wrapper of the `pytorch_geometric.explain.Explainer.visualize_feature_importance` method

        Args:
            path (os.PathLike | None, optional): Path to save the feature importance graph. Defaults to None.
            top_k (int | None, optional): The number of features to include in the graph. Defaults to None.
        """
        feat_labels = None
        if isinstance(self.clinical_data, pd.DataFrame):
            feat_labels = self.clinical_data.columns.to_list()

        self.expl.visualize_feature_importance(
            str(path) if path != None else path, top_k=top_k, feat_labels=feat_labels  # type: ignore
        )
