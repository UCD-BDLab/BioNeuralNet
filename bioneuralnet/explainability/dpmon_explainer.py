# explainability/dpmon_explainer.py
# Kyle Rohn / Justin Hoang

import torch
from torch import nn
from torch.types import FileLike
import torch.nn.functional as F

from pprint import pformat

from typing import Literal, Dict, List

from torch_geometric.data import Data
from torch_geometric.explain import (
    Explainer,
    ExplainerAlgorithm,
    ModelConfig,
    GNNExplainer,
)


import pandas as pd
import numpy as np

from bioneuralnet.downstream_task import DPMON
from bioneuralnet.downstream_task.dpmon import (
    NeuralNetwork,
    prepare_node_features,
    slice_omics_datasets,
    setup_device,
)
from bioneuralnet.utils import get_logger


class NeuralNetworkWrapper(nn.Module):
    """A wrapper class for formatting DPMON Neural Network IO in a form pytorch_geometric requires"""

    def __init__(self, nn: NeuralNetwork):
        """Create NeuralNetworkWrapper instance.

        Args:
            nn (NeuralNetwork): The underlying DPMON neural network model to wrap.
        """

        super(NeuralNetworkWrapper, self).__init__()

        self.nn = nn
        self.eval()

    def forward(self, x, edge_index, train_features, **kwargs):
        """Forward pass through the wrapped neural network.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, num_features].
            edge_index (torch.Tensor): Graph edge indices of shape [2, num_edges].
            train_features (torch.Tensor): Training features for the omics data.
            **kwargs: Additional arguments passed to the Data constructor.

        Returns:
            torch.Tensor: Model predictions.
        """
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

        Args:
            f (FileLike): The file object or path to a saved model trained with DPMON
            dpmon (DPMON): the DPMON object which trained the model
            weights_only (bool, optional): Load model weights only (you probably want true). Defaults to True.
        """
        # Load weights of model
        self.logger = get_logger(__name__)
        self.logger.info("Initializing DPMONExplainer")

        model_weights = torch.load(f, weights_only=weights_only)
        self.logger.debug(f"Model weights loaded from {f}")

        # setup device
        device = setup_device(dpmon.gpu, dpmon.cuda)
        self.logger.info(f"Using device: {device}")

        self.omics_dataset = slice_omics_datasets(
            dpmon.combined_omics, dpmon.adjacency_matrix, dpmon.phenotype_col
        )[0]
        self.logger.info(f"Loaded omics dataset with shape: {self.omics_dataset.shape}")
        self.logger.debug(
            f"Phenotype distribution: {self.omics_dataset['phenotype'].value_counts().to_dict()}"
        )

        self.omics_network_tg = prepare_node_features(
            adjacency_matrix=dpmon.adjacency_matrix,
            omics_datasets=[self.omics_dataset],
            clinical_data=dpmon.clinical_data,
            phenotype_col="phenotype",
        )[0]
        if (
            self.omics_network_tg.x is not None
            and self.omics_network_tg.edge_index is not None
        ):
            self.logger.info(
                f"Graph prepared with {self.omics_network_tg.x.shape[0]} nodes and {self.omics_network_tg.edge_index.shape[1]} edges"
            )
        else:
            self.logger.warning("Graph preparation may have incomplete data")

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
        self.logger.info(
            f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters"
        )

        self.train_features = torch.FloatTensor(
            self.omics_dataset.drop(["phenotype"], axis=1).values
        ).to(device)
        self.logger.debug(
            f"Training features tensor shape: {self.train_features.shape}"
        )

        # self.target = torch.Tensor(self.omics_dataset["phenotype"].values)

        self.model = NeuralNetworkWrapper(model)
        self.logger.info("DPMONExplainer initialization complete")

    def explain(
        self,
        algorithm: ExplainerAlgorithm,
        mode: Literal[
            "regression", "binary_classification", "multiclass_classification"
        ],
        explanation_type: Literal["model", "phenomenon"],
        node_mask_type: Literal["object", "common_attributes", "attributes"] | None,
        edge_mask_type: Literal["object", "common_attributes", "attributes"] | None,
        task_level: Literal["edge", "node", "graph"] = "graph",
        return_type: Literal["raw", "log_probs", "probs"] = "raw",
    ) -> None:
        """Generate explanations for the DPMON model using a specified explainer algorithm.

        This is a general-purpose method for advanced users. Note that ExplainerAlgorithm instantiation
        (such as for PGExplainer) must occur outside this method.

        Args:
            algorithm (ExplainerAlgorithm): The pytorch_geometric.explain.ExplainerAlgorithm instance to use for explanation generation.
            mode (Literal["regression", "binary_classification", "multiclass_classification"]): The task mode indicating the type of prediction task.
            explanation_type (Literal["model", "phenomenon"]): Whether to explain model predictions ("model") or the underlying phenomenon ("phenomenon").
            node_mask_type (Literal["object", "common_attributes", "attributes"] | None): Type of node masking strategy. None disables node masking.
            edge_mask_type (Literal["object", "common_attributes", "attributes"] | None): Type of edge masking strategy. None disables edge masking.
            task_level (Literal["edge", "node", "graph"], optional): The granularity level of explanation (edge, node, or graph-level). Defaults to "graph".
            return_type (Literal["raw", "log_probs", "probs"], optional): The format of model outputs (raw logits, log probabilities, or probabilities). Defaults to "raw".
        """
        self.logger.info(
            f"Starting explanation generation with algorithm: {algorithm.__class__.__name__}"
        )
        self.logger.debug(
            f"Explanation config - mode: {mode}, task_level: {task_level}, return_type: {return_type}"
        )

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
        self.logger.debug("Explainer instance created")

        if self.omics_network_tg.x != None and self.omics_network_tg.edge_index != None:
            self.logger.info("Generating explanations for the entire network")
            self.expl = self.explainer(
                **self.omics_network_tg.to_dict(), train_features=self.train_features
            )
            self.logger.info("Explanation generation complete")
        else:
            self.logger.warning("Network data missing (x or edge_index is None)")

    def generate_omics_summary(
        self,
        mode: Literal[
            "regression", "binary_classification", "multiclass_classification"
        ],
        index: int | None = None,
        patient_id: str | None = None,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: Literal["raw", "log_probs", "probs"] = "raw",
        include_correct: bool = False,
    ) -> pd.DataFrame:
        """Generate omics feature importance summaries using GNN explanations.

        This method generates feature importance explanations for either a single patient or
        aggregated across all patients grouped by target class. For single patient analysis,
        returns per-feature importance scores. For multi-patient analysis, returns average
        importance scores per target class (only considering correct predictions).

        Args:
            mode (Literal["regression", "binary_classification", "multiclass_classification"]): The task mode for explanation generation.
            index (int | None, optional): The integer index of the patient to explain (0-based). Defaults to None.
                If provided with patient_id, patient_id takes precedence.
            patient_id (str | None, optional): The patient identifier (row name in the omics dataset). Defaults to None.
                Either index or patient_id should be provided for single-patient analysis.
            epochs (int, optional): Number of training epochs for GNNExplainer. Defaults to 100.
            lr (float, optional): Learning rate for GNNExplainer optimization. Defaults to 0.01.
            return_type (Literal["raw", "log_probs", "probs"], optional): The format of model outputs. Defaults to "raw".
            include_correct (bool, optional): Controls which samples to explain in aggregate mode. Defaults to False.
                If False, only samples with correct predictions are explained (for more reliable feature importance).
                If True, all samples are explained regardless of prediction correctness (for comprehensive analysis including errors).
                This parameter is only used in aggregate (dataset-level) mode when neither index nor patient_id is provided.


        Returns:
            pd.DataFrame: If index or patient_id provided, returns a pd.DataFrame with:
                - Index: omics feature names (excluding phenotype)
                - Columns: patient identifier
                - Values: feature importance scores sorted in descending order

                If neither index nor patient_id provided, returns a pd.DataFrame with:
                - Index: omics feature names (excluding phenotype)
                - Columns: target class labels
                - Values: average importance scores (aggregated from samples based on include_correct parameter)

        Raises:
            AttributeError: If patient_id matches multiple rows in the dataset.
        """
        explainer_args = {
            "algorithm": GNNExplainer(epochs=epochs, lr=lr),
            "explanation_type": "model",
            "node_mask_type": "object",
            "model_config": {
                "mode": mode,
                "task_level": "graph",
                "return_type": return_type,
            },
        }

        explainer = Explainer(self.model, **explainer_args)

        self.logger.info(
            f"Created pytorch_geometric.Explainer object with params:\n{pformat(explainer_args)}"
        )
        self.logger.info(
            f"Dataset size: {len(self.omics_dataset)} samples, {len(self.omics_dataset.columns)-1} features"
        )
        
        self.faithfulness_scores = []

        if index != None or patient_id != None:

            if index != None and patient_id != None:
                # Verify that index and patient_id refer to the same patient
                index_patient = self.omics_dataset.index[index]
                patient_index = self.omics_dataset.index.get_loc(patient_id)
                if index != patient_index or index_patient != patient_id:
                    self.logger.warning(
                        f"index and patient_id mismatch: index={index} (patient_id={index_patient}) vs patient_id={patient_id} (index={patient_index})"
                    )
                    self.logger.info("Using patient_id to determine the sample")

            if patient_id is not None:
                _patient_id = patient_id
                # Resolve patient_id to index, catching KeyError if patient not found
                try:
                    _index_result = self.omics_dataset.index.get_loc(patient_id)
                    if not isinstance(_index_result, int):
                        self.logger.error(
                            f"Patient with id {patient_id} matched multiple rows in dataset"
                        )
                        raise AttributeError(
                            f"Patient with id {patient_id} matched multiple rows in dataset"
                        )
                    _index = _index_result
                except KeyError:
                    self.logger.error(
                        f"Patient with id {patient_id} not found in dataset"
                    )
                    raise AttributeError(
                        f"Patient with id {patient_id} not found in dataset"
                    )
            else:
                # index is not None here due to the outer if condition
                assert index is not None
                _index = index
                _patient_id = self.omics_dataset.index[_index]

            self.logger.info(
                f"Generating summary for single patient - index: {_index}, patient_id: {_patient_id}"
            )
            self.logger.debug(f"Patient index resolved to: {_index}")

            train_features_i = self.train_features[_index : _index + 1]
            pred = None
            with torch.no_grad():
                pred = torch.argmax(
                    self.model(
                        **self.omics_network_tg.to_dict(),
                        train_features=train_features_i,
                    )
                ).item()

            # Generate explanations for a patient at a specific index
            explanation = explainer(
                **self.omics_network_tg.to_dict(), train_features=train_features_i
            )

            omics = list(self.omics_dataset.columns)
            omics.remove("phenotype")
            target = self.omics_dataset["phenotype"].loc[_patient_id].item()

            self.logger.debug(f"Patient target: {target}, Prediction: {pred}")

            patient_data = {_patient_id: [val.item() for val in explanation.node_mask]}

            out = pd.DataFrame(data=patient_data, index=omics)

            self.logger.info("Single-patient explanation complete")
            return out

        else:
            # Generate average explanations for each target value
            self.logger.info(
                "Generating aggregate explanations for all patients by target class"
            )

            trues = self.omics_dataset["phenotype"]
            targets = sorted(trues.unique())
            trues = trues.values

            self.logger.debug(f"Target classes: {targets}")

            # Get accuracy per prediction
            logits = None
            with torch.no_grad():
                logits = self.model(
                    **self.omics_network_tg.to_dict(),
                    train_features=self.train_features,
                )
            preds = [torch.argmax(pred).item() for pred in logits]
            assert len(preds) == len(trues)

            accuracy = sum(1 for t, p in zip(trues, preds) if t == p) / len(trues)
            self.logger.info(f"Model accuracy on dataset: {accuracy:.4f}")

            cols = {}
            counts = {}
            for t in targets:
                cols[t] = np.zeros(self.omics_dataset.shape[1] - 1)
                counts[t] = 0

            correct = 0

            for idx, id in enumerate(self.omics_dataset.index):
                if include_correct or (trues[idx] == preds[idx]):
                    train_features_i = self.train_features[idx : idx + 1].clone().detach().requires_grad_(True)
                    expl = explainer(
                        **self.omics_network_tg.to_dict(),
                        train_features=train_features_i,
                    )
                    self.logger.info(
                        f"Generating explanations for patient {id} ({idx+1}/{len(self.omics_dataset.index)})"
                    )
                    correct += 1
                    cols[trues[idx]] += expl.node_mask.cpu().numpy().flatten()
                    counts[trues[idx]] += 1
                    
                    logit = logits[0, preds[idx]]
                    logit.backward()
                    grad = train_features_i.grad
                    
                    if grad != None:
                        score = F.cosine_similarity(grad.abs(), expl.node_mask)
                        self.logger.info(f"Faithfunless score for patient {id}: {score}")
                        self.faithfulness_scores.append(score)
                    
                else:
                    self.logger.info(
                        f"Skipping patient {id}: true={trues[idx]}, pred={preds[idx]}"
                    )

            if include_correct:
                self.logger.info(
                    f"Processed {correct} total patients (including incorrect predictions)"
                )
            else:
                self.logger.info(f"Processed {correct} correctly predicted patients")

            # Average the importance scores by class
            for t in targets:
                if counts[t] > 0:
                    cols[t] = cols[t] / counts[t]
                    self.logger.debug(
                        f"Averaged importance for class {t}: {counts[t]} samples"
                    )

            omics = list(self.omics_dataset.columns)
            omics.remove("phenotype")

            result = pd.DataFrame(data=cols, index=omics)
            self.logger.info(
                f"Generated aggregate explanation dataframe with shape: {result.shape}"
            )

            return result

    