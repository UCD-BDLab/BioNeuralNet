import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ray.air import session

from ..utils.logger import get_logger

class GraphEmbedding:
    """
    GraphEmbedding class for Integrating Network Embeddings into Omics Data.
    Nodes represent omics features, labeled by correlation to a phenotype. 
    GNNs Embeddings from GNNEmbeddings class are reintegrated into the original omics data for improved predictive performance.

    Attributes:
    
        omics_data : pd.DataFrame
        embeddings : pd.DataFrame
        phenotype_data : Optional[pd.DataFrame], default=None
        phenotype_col : str, default="phenotype"
        reduce_method : str, default="PCA"
        tune : Optional[bool], default=False
    """

    def __init__(
        self,
        omics_data: pd.DataFrame,
        embeddings: pd.DataFrame,
        phenotype_data: Optional[pd.DataFrame] = None,
        phenotype_col: str = "phenotype",
        reduce_method: str = "PCA",
        tune: Optional[bool] = False,
    ):
        """
        Initializes the GraphEmbedding instance.

        Parameters:
            omics_data : pd.DataFrame
            embeddings : Optional[pd.DataFrame], default=None
            reduce_method : str, optional
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing GraphEmbedding with provided data inputs.")

        if omics_data is None or omics_data.empty:
            raise ValueError("Omics data must be non-empty.")

        if embeddings is None or embeddings.empty:
            self.logger.info(
                "No embeddings provided, please review documentation to see how to generate embeddings."
            )
        if not isinstance(embeddings, pd.DataFrame):
            raise ValueError("Embeddings must be provided as a pandas DataFrame.")
        
        if tune == True and phenotype_data is None:
            raise ValueError(
                "Phenotype data must be provided for classification-based tuning."
            )
        
        self.omics_data = omics_data
        self.embeddings = embeddings if embeddings is not None else pd.DataFrame()
        self.phenotype_data = phenotype_data
        self.phenotype_col = phenotype_col
        self.reduce_method = reduce_method.upper()
        self.tune = tune

        embeddings_features = set(self.embeddings.index)
        omics_features = set(self.omics_data.columns)
        if len(embeddings_features) != len(omics_features):
            raise ValueError(
                f"Number of features in embeddings and omics data do not match.\n"
                f"Embeddings: {self.embeddings.shape} and Omics: {self.omics_data.shape}"
            )
        common_features = embeddings_features.intersection(omics_features)
        if len(common_features) == 0:
            raise ValueError(
                f"No common features found between the embeddings and omics data.\n"
                f"Embeddings: {self.embeddings.shape} and Omics: {self.omics_data.shape}"
            )
        self.logger.info(
            f"Found {len(common_features)} common features between network and omics data."
        )

    def run(self) -> pd.DataFrame:
        """
        If tune=True, perform classification-based tuning (if phenotype_data provided),
        else fallback to a default embedding reduction method.
        """
        self.logger.info("Starting Subject Representation workflow.")

        if self.embeddings.empty:
            self.logger.warning(
                "No embeddings provided. Please generate emebeddings using GNNEmbeddings class.\nReturning original omics_data."
            )
            return self.omics_data

        try:
            if self.tune:
                best_config = self._run_tuning()
                self.logger.info(f"Best tuning config selected: {best_config}")

                reduced = self._reduce_embeddings(
                    method=best_config["method"],
                    pca_dim=best_config.get("pca_dim", 1),
                    ae_params=best_config.get(
                        "ae_params", {"epochs": 16, "hidden_dim": 8}
                    ),
                )
            else:
                reduced = self._reduce_embeddings(
                    method=self.reduce_method,
                    pca_dim=2,
                )

            if reduced.empty:
                self.logger.warning(
                    "Reduced embeddings are empty. Returning original omics_data."
                )
                return self.omics_data

            enhanced_omics_data = self._integrate_embeddings(reduced)
            self.logger.info(
                f"Subject Representation completed successfully. Final shape: {enhanced_omics_data.shape}"
            )
            return enhanced_omics_data

        except Exception as e:
            self.logger.error(f"Error in Subject Representation workflow: {e}")
            raise

    def _reduce_embeddings(
        self, method: str, pca_dim: int = 2, ae_params: Dict[str, Any] = None
    ) -> pd.Series:
        
        self.logger.info(f"Reducing embeddings to {pca_dim} using method='{method}'.")
        
        if self.embeddings.empty:
            raise ValueError("Embeddings DataFrame is empty.")
        
        if method == "PCA":
            self.logger.info(f"Applying PCA with n_components={pca_dim}.")
            pca = PCA(n_components=pca_dim)
            pcs = pca.fit_transform(self.embeddings)
            
            if pca_dim == 1:
                reduced_embedding = pd.Series(
                    pcs.flatten(), index=self.embeddings.index, name="PC1"
                )
            else:
                reduced_embedding = pd.Series(
                    pcs.mean(axis=1), index=self.embeddings.index, name="PC_mean"
                )
            self.logger.info(
                "Captured variance ratio: %.2f" % pca.explained_variance_ratio_[0]
            )
            self.logger.info("PCA reduction completed.")

        elif method == "AE":
            self.logger.info("Using Autoencoder for reduction.")
            if ae_params is None:
                ae_params = {"epochs": 16, "hidden_dim": 8, "lr": 1e-3}
            input_dim = self.embeddings.shape[1]
            X = torch.tensor(self.embeddings.values, dtype=torch.float)
            model = SimpleAE(
                input_dim=input_dim,
                hidden_dim=ae_params.get("hidden_dim", 8),
                compressed_dim=1,
            )
            optimizer = optim.Adam(model.parameters(), lr=ae_params.get("lr", 1e-3))
            loss_fn = nn.MSELoss()
            model.train()
            epochs = ae_params.get("epochs", 16)
            for epoch in range(epochs):
                optimizer.zero_grad()
                z, recon = model(X)
                loss = loss_fn(recon, X)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % max(1, epochs // 5) == 0:
                    self.logger.info(f"AE Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                z, _ = model(X)
            reduced_embedding = pd.Series(
                z.squeeze().numpy(), index=self.embeddings.index, name="AE"
            )
            self.logger.info("Autoencoder reduction completed.")
        else:
            self.logger.error(f"Unsupported reduction method: {method}")
            raise ValueError(f"Unsupported reduction method: {method}")

        reduced_embedding = (reduced_embedding - reduced_embedding.mean()) / (
            reduced_embedding.std() + 1e-8
        )
        self.logger.info("Reduced embedding normalized.")
        return reduced_embedding

    def _integrate_embeddings(
        self, reduced: pd.Series, method="multiply"
    ) -> pd.DataFrame:
        self.logger.info(f"Integrating embeddings using method='{method}'.")

        common = list(set(self.omics_data.columns).intersection(set(reduced.index)))
        if not common:
            raise ValueError("No common features between omics data and embeddings.")

        enhanced_omics = self.omics_data.copy()

        if method == "multiply":
            for feature in common:
                enhanced_omics[feature] = enhanced_omics[feature] * reduced[feature]
            enhanced_omics = enhanced_omics[common]
            self.logger.info(
                "Integration using multiplication completed. (Columns overwritten)"
            )
        elif method == "concatenate":
            enhanced_features = pd.DataFrame(
                np.tile(reduced.values, (self.omics_data.shape[0], 1)),
                index=self.omics_data.index,
                columns=[f"{feat}_embed" for feat in reduced.index],
            )
            enhanced_omics = pd.concat([enhanced_omics, enhanced_features], axis=1)
            self.logger.info("Integration using concatenation completed.")

        elif method == "weighted":
            weights = np.abs(reduced.values) / (np.sum(np.abs(reduced.values)) + 1e-8)
            for feature, weight in zip(common, weights):
                enhanced_omics[f"{feature}_embed"] = (
                    enhanced_omics[feature] + weight * reduced[feature]
                )
            self.logger.info("Integration using weighted sum completed.")
        else:
            raise ValueError(f"Unknown integration method: {method}")

        self.logger.info(f"Final Enhanced Omics Shape: {enhanced_omics.shape}")
        return enhanced_omics

    def _run_tuning(self) -> Dict[str, Any]:
        """
        Classification-based tuning
        """

        self.logger.info("Running classification-based tuning for GraphEmbedding.")
        return self._run_classification_tuning()

    def _run_classification_tuning(self) -> Dict[str, Any]:
        search_config = {
            "method": tune.choice(["PCA", "AE"]),
            "pca_dim": tune.choice([1, 2, 3]),
            "ae_params": tune.choice(
                [
                    {"epochs": 64, "hidden_dim": 128},
                    {"epochs": 128, "hidden_dim": 16},
                    {"epochs": 256, "hidden_dim": 8},
                    {"epochs": 128, "hidden_dim": 16},
                    {"epochs": 256, "hidden_dim": 4},
                    {"epochs": 512, "hidden_dim": 4},
                ]
            ),
            "integration_method": tune.choice(["multiply"]),
        }

        def tune_helper(config):
            reduced = self._reduce_embeddings(
                config["method"], config["pca_dim"], config["ae_params"]
            )
            if config["integration_method"] == "multiply":
                enhanced = self._integrate_embeddings(reduced, method="multiply")
            else:
                raise ValueError("Unknown integration method")

            common_samples = enhanced.index.intersection(self.phenotype_data.index)
            X = enhanced.loc[common_samples].values
            y = self.phenotype_data.loc[common_samples, self.phenotype_col].astype(int).values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            session.report({"accuracy": acc})

        scheduler = ASHAScheduler(metric="accuracy", mode="max", grace_period=1, reduction_factor=2)
        reporter = CLIReporter(metric_columns=["accuracy", "training_iteration"])

        def short_dirname_creator(trial):
            return f"_{trial.trial_id}"

        analysis = tune.run(
            tune_helper,
            config=search_config,
            num_samples=10,
            verbose=0,
            scheduler=scheduler,
            progress_reporter=reporter,
            trial_dirname_creator=short_dirname_creator,
            name="tune",
        )

        best_trial = analysis.get_best_trial("accuracy", "max", "last")
        self.logger.info(f"Best trial config: {best_trial.config}")
        self.logger.info(f"Best trial final accuracy: {best_trial.last_result['accuracy']}")

        best_params_file = "Graph_embedding_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(best_trial.config, f, indent=4)
        self.logger.info(f"Best Graph Embedding parameters saved to {best_params_file}")
        
        return best_trial.config


class SimpleAE(nn.Module):
    def __init__(self, input_dim, hidden_dim: int = 8, compressed_dim: int = 1):
        """
        Taking reference from DPMON, I modified SimpleAE class to deepen the network architecture.
        
        Parameters:
            input_dim (int): Dimensionality of the input.
            hidden_dim (int): Size of the first hidden layer. The second hidden layer is set to half this size.
            compressed_dim (int): Dimensionality of the latent representation.
        """
        super(SimpleAE, self).__init__()
        hidden1 = hidden_dim
        hidden2 = max(1, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, compressed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

