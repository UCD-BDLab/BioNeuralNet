import pandas as pd
import os
import json
import tempfile
from typing import Optional, Dict, Any, List, Union, Sequence
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:
    raise ImportError(
        "SubjectRepresentation requires PyTorch. "
        "Please install it by following the instructions at: "
        "https://bioneuralnet.readthedocs.io/en/latest/installation.html"
    )

from ray import tune
from ray.tune import TuneError
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from bioneuralnet.utils import get_logger

class SubjectRepresentation:
    """SubjectRepresentation Class for Integrating Network Embeddings into Omics Data.

    This class integrates network-derived embeddings with raw omics data to create enriched subject-level profiles. It supports dimensionality reduction of embeddings (via Autoencoders or other methods) and subsequent fusion with original omics features.

    Attributes:
        omics_data (pd.DataFrame): DataFrame of omics features (columns).
        embeddings (pd.DataFrame): DataFrame with embeddings (indexed by feature names).
        phenotype_data (Optional[pd.DataFrame]): Optional DataFrame with phenotype labels.
        phenotype_col (str): Name of the phenotype column.
        reduce_method (str): Method used for dimensionality reduction (e.g., "AE").
        seed (Optional[int]): Random seed for reproducibility.
        tune (bool): Whether to run hyperparameter tuning.
        output_dir (Path): Directory where results are written.
    """

    def __init__(
        self,
        omics_data: pd.DataFrame,
        embeddings: pd.DataFrame,
        phenotype_data: Optional[pd.DataFrame] = None,
        phenotype_col: str = "phenotype",
        reduce_method: str = "AE",
        seed: Optional[int] = None,
        tune: Optional[bool] = False,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initializes the SubjectRepresentation instance.

        Args:
            omics_data (pd.DataFrame): DataFrame of omics features (columns).
            embeddings (pd.DataFrame): DataFrame with embeddings (indexed by feature names).
            phenotype_data (pd.DataFrame | None): Optional DataFrame with phenotype labels.
            phenotype_col (str): Name of the phenotype column. Default is "phenotype".
            reduce_method (str): Dimensionality reduction method (e.g., "AE"). Default is "AE".
            seed (int | None): Random seed for reproducibility.
            tune (bool | None): Whether to run hyperparameter tuning. Default is False.
            output_dir (str | Path | None): Directory to write results. If None, a temp directory is used.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing SubjectRepresentation with provided data inputs.")

        if omics_data is None or omics_data.empty:
            raise ValueError("Omics data must be non-empty.")

        if embeddings is None:
            self.logger.warning("No embeddings provided, please review documentation to see how to generate embeddings.")
            raise ValueError("Embeddings must be non-empty.")

        if not isinstance(embeddings, pd.DataFrame):
            raise ValueError("Embeddings must be provided as a pandas DataFrame.")

        if embeddings.empty:
            self.logger.warning("No embeddings provided, please review documentation to see how to generate embeddings.")
            raise ValueError("Embeddings must be non-empty.")

        if tune and phenotype_data is None:
            raise ValueError("Phenotype data must be provided for classification-based tuning.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.seed = seed
        self.logger.info(f"Seed set to {self.seed}.")

        self.omics_data = omics_data.copy(deep=True)
        self.embeddings = embeddings.copy(deep=True)
        self.phenotype_data = phenotype_data
        self.phenotype_col = phenotype_col
        self.reduce_method = reduce_method.upper()
        self.tune = tune

        # match the embedding index
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
        self.logger.info(f"Found {len(common_features)} common features between network and omics data.")

        # output directory
        if output_dir is None:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = Path(self.temp_dir_obj.name)
            self.logger.info(f"No output_dir provided; using temporary directory: {self.output_dir}")
        else:
            self.output_dir = Path(output_dir)
            self.logger.info(f"Output directory set to: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Executes the Subject Representation workflow.

        If tuning is enabled, runs hyperparameter tuning and uses the best config to reduce embeddings. Otherwise, uses the default reduction method.

        Returns:
            pd.DataFrame: Enhanced omics data as a DataFrame.
        """
        self.logger.info("Starting Subject Representation workflow.")

        if self.embeddings.empty:
            self.logger.warning(
                "No embeddings provided. Please generate embeddings using GNNEmbeddings class.\nReturning original omics_data."
            )
            return self.omics_data

        try:
            if self.tune:
                best_config = self._run_tuning()
                self.logger.info(f"Best tuning config selected: {best_config}")
                ae_params = best_config.get("ae_params", {"epochs": 16, "hidden_dim": 8, "lr": 1e-3, "dropout": 0.2, "activation": "relu"})
                reduced = self._reduce_embeddings(method=best_config["method"], ae_params=ae_params, compressed_dim=best_config["compressed_dim"])
                enhanced_omics_data = self._integrate_embeddings(reduced=reduced, method=best_config["integration_method"], alpha=best_config["alpha"], beta=best_config["beta"])

            else:
                method = self.reduce_method.upper()
                ae_params_def = {"epochs": 16, "hidden_dim": 8, "lr": 1e-3, "dropout": 0.2, "activation": "relu"}
                reduced = self._reduce_embeddings(method=method, ae_params=ae_params_def, compressed_dim=2)
                enhanced_omics_data = self._integrate_embeddings(reduced, method="multiply", alpha=1.0, beta=0.8)

            if enhanced_omics_data.empty:
                self.logger.warning("Enhanced omics data is empty. Returning original omics_data.")
                return self.omics_data

            self.logger.info(f"Subject Representation completed successfully. Final shape: {enhanced_omics_data.shape}")
            return enhanced_omics_data

        except Exception as e:
            self.logger.error(f"Error in Subject Representation workflow: {e}")
            raise

    def _reduce_embeddings(self, method: str, ae_params: Optional[dict[Any, Any]] = None, compressed_dim: int = 2) -> pd.DataFrame:
        """Reduces the dimensionality of embeddings using PCA or AutoEncoder.

        Args:

            method (str): Reduction method ("PCA" or "AE").
            ae_params (dict | None): AutoEncoder hyperparameters (epochs, hidden_dim, lr, dropout).
            compressed_dim (int): Target number of dimensions.

        Returns:

            pd.DataFrame: Reduced embeddings with `compressed_dim` columns.

        """
        self.logger.info(f"Reducing embeddings using method='{method}' with compressed_dim={compressed_dim}.")

        if self.embeddings.empty:
            raise ValueError("Embeddings DataFrame is empty.")

        method = method.lower()
        if method == "pca":
            pca = PCA(n_components=compressed_dim)
            pcs = pca.fit_transform(self.embeddings)
            columns = []
            for i in range(compressed_dim):
                columns.append(f"PC{i+1}")

            reduced_df = pd.DataFrame(pcs, index=self.embeddings.index, columns=columns)
            self.logger.info(f"PCA reduction completed. Variance explained: {pca.explained_variance_ratio_.sum():.4f}")

        elif method in ["ae"]:
            self.logger.info("Using autoencoder for reduction.")
            ae_params = ae_params or {
                "epochs": 64,
                "hidden_dim": 8,
                "lr": 1e-3,
                "dropout": 0.2,
                "activation": "relu",
            }
            X = torch.tensor(self.embeddings.values, dtype=torch.float)

            model = AutoEncoder(
                input_dim=self.embeddings.shape[1],
                hidden_dims=ae_params["hidden_dim"],
                compressed_dim=compressed_dim,
                dropout=ae_params["dropout"],
                activation=ae_params["activation"]
            )
            optimizer = optim.Adam(model.parameters(), lr=ae_params["lr"])
            loss_fn = nn.MSELoss()

            model.train()
            for epoch in range(ae_params["epochs"]):
                optimizer.zero_grad()
                z, recon = model(X)
                loss = loss_fn(recon, X)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % max(1, ae_params["epochs"] // 5) == 0:
                    self.logger.info(f"AE Epoch {epoch+1}/{ae_params['epochs']} - Loss: {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                z, _ = model(X)

            z_np = z.detach().cpu().numpy()
            if z_np.ndim == 1:
                z_np = z_np.reshape(-1, 1)
            columns = []
            for i in range(compressed_dim):
                columns.append(f"AE_{i+1}")

            reduced_df = pd.DataFrame(z_np, index=self.embeddings.index, columns=columns)
            self.logger.info("Autoencoder reduction completed")

        else:
            raise ValueError(f"Unsupported reduction method:{method}")

        return reduced_df

    def _integrate_embeddings(self, reduced: pd.DataFrame, method="multiply", alpha=2.0, beta=0.5) -> pd.DataFrame:
        """Integrates reduced embeddings into omics data using a weighted multiplicative approach.

        Formula: `enhanced = beta * raw + (1 - beta) * (alpha * normalized_weight * raw)`

        Args:
            reduced (pd.DataFrame | pd.Series): Reduced embedding weights.
            method (str): Integration strategy (currently only "multiply").
            alpha (float): Scaling factor for the weighted term.
            beta (float): Weight applied to the original raw data (0 to 1).

        Returns:
            pd.DataFrame: Enhanced omics data.
        """
        if not isinstance(reduced, (pd.DataFrame, pd.Series)):
            raise ValueError("Reduced embeddings must be a pandas DataFrame or Series.")

        missing_features = set(self.omics_data.columns) - set(reduced.index)
        if missing_features:
            self.logger.warning(f"Missing {len(missing_features)} features in reduced embeddings: {list(missing_features)[:5]}")

        if method == "multiply":
            # the reduced embeddings to match the omics data columns.
            if not reduced.index.equals(self.omics_data.columns):
                self.logger.info("Aligning reduced embeddings index with omics_data columns.")
                reduced = reduced.reindex(self.omics_data.columns)

            # normalize the embeddings and compute a weight series
            if isinstance(reduced, pd.DataFrame):
                for col in reduced.columns:
                    reduced[col] = (reduced[col] - reduced[col].min()) / (reduced[col].max() - reduced[col].min())
                weight_series = reduced.mean(axis=1)
            elif isinstance(reduced, pd.Series):
                weight_series = (reduced - reduced.min()) / (reduced.max() - reduced.min())

            weight_series = weight_series.fillna(0)
            ranks = weight_series.rank(method="average")
            scaled_ranks = 2 * (ranks - ranks.min()) / (ranks.max() - ranks.min()) - 1

            # look for duplicate feature names.
            if not scaled_ranks.index.is_unique:
                self.logger.error("Duplicate feature names detected in the reduced embeddings index.")

            # re-index to ensure we have a weight per omics_data feature.
            feature_weights = scaled_ranks.reindex(self.omics_data.columns,fill_value=0.0)

            enhanced = self.omics_data.copy()
            for feature in self.omics_data.columns:
                weight_val = feature_weights.get(feature)
                # checking weight_val is series or scalar
                if isinstance(weight_val, pd.Series):
                    weight_val = weight_val.iloc[0]
                if pd.notnull(weight_val):
                    enhanced[feature] = beta * enhanced[feature] + (1 - beta) * (alpha * weight_val * enhanced[feature])
                else:
                    self.logger.warning(f"Feature {feature} not found in the reduced weights.")
        else:
            #Curently only supoort one method but left the parameter in case we supp0ort more in the future.
            raise ValueError(f"Unknown integration method: {method}.")

        self.logger.info(f"Final Enhanced Omics Shape: {enhanced.shape}")
        return enhanced

    def _run_tuning(self) -> Dict[str, Any]:
        """Orchestrates the hyperparameter tuning process.

        Returns:
            dict: The best configuration found by Ray Tune.
        """
        self.logger.info("Running tuning for SubjectRepresentation.")
        return self._run_classification_tuning()

    def _run_classification_tuning(self) -> Dict[str, Any]:
        """Executes Ray Tune to find optimal reduction and integration parameters.

        Uses RandomForest (Classifier or Regressor) to evaluate the quality of the
        enhanced subject representations against the phenotype.

        Returns:
            dict: Best hyperparameter configuration.
        """
        search_config = {
            "method": tune.choice(["PCA", "AE"]),
            "compressed_dim": tune.choice([1, 2, 3, 4]),
            "ae_params": {
                "epochs": tune.choice([64, 128, 256, 512, 1024]),
                "hidden_dim": tune.choice([16, 32, 64, 128, 256, 512]),
                "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
                "lr": tune.choice([1e-3, 5e-4, 1e-4, 1e-5]),
                "activation": tune.choice(["relu", "tanh", "sigmoid"]),
            },
            "integration_method": tune.choice(["multiply"]),
            "alpha": tune.choice([1.5, 2.0, 2.5, 3.0]),
            "beta": tune.choice([0.1, 0.3, 0.5, 0.7]),
        }

        def tune_helper(config):
            try:
                method = config["method"].upper()
                ae_params = config["ae_params"]
                alpha = config["alpha"]
                beta = config["beta"]
                compressed_dim = config["compressed_dim"]

                reduced = self._reduce_embeddings(method=method, compressed_dim=compressed_dim,ae_params=ae_params)
                enhanced = self._integrate_embeddings(reduced, method=config["integration_method"], alpha=alpha, beta=beta)
                common_index = enhanced.index.intersection(self.phenotype_data.index)

                X = enhanced.loc[common_index].values
                y = self.phenotype_data.loc[common_index, self.phenotype_col]

                is_classification = y.dtype != float and y.nunique() <= 20
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                if is_classification:
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train.astype(int))
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                else:
                    model = RandomForestRegressor()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    score = -mse

                tune.report({"score": score})

            except Exception as e:
                self.logger.error(f"Tuning trial failed: {e}")
                import traceback
                traceback.print_exc()
                tune.report({"score": 0.0})

        scheduler = ASHAScheduler(metric="score", mode="max", grace_period=1, reduction_factor=2)
        reporter = CLIReporter(metric_columns=["score", "training_iteration"])

        def short_dirname_creator(trial):
            return f"_{trial.trial_id}"

        resources = {"cpu": 1, "gpu": 1} if torch.cuda.is_available() else {"cpu": 1, "gpu": 0}

        try:
            analysis = tune.run(
                tune_helper,
                config=search_config,
                num_samples=20,
                scheduler=scheduler,
                progress_reporter=reporter,
                storage_path=os.path.expanduser("~/sr"),
                trial_dirname_creator=short_dirname_creator,
                resources_per_trial=resources,
                name="t_sr",
                verbose=1,
            )
        except TuneError as e:
            self.logger.warning(f"Tuning completed with failures: {e}")
            analysis = None
            if len(e.args) > 1 and hasattr(e.args[1], "get_best_trial"):
                analysis = e.args[1]

        best_trial = None
        best_score = 0.0

        if analysis and hasattr(analysis, "get_best_trial"):
            try:
                best_trial = analysis.get_best_trial("score", "max", "last")
                best_score = best_trial.last_result.get("score", 0.0)
            except Exception as e:
                self.logger.error(f"Could not retrieve best trial details: {e}")
        else:
            self.logger.error("Analysis object is invalid or missing get_best_trial().")

        self.logger.info(f"Best trial final score: {best_score}")

        if best_trial:
            timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
            best_params_file = Path(self.output_dir) / f"Subjects_tune_{timestamp}.json"

            with open(best_params_file, "w") as f:
                json.dump(best_trial.config, f, indent=4)
            self.logger.info(f"Best Graph Embedding parameters saved to {best_params_file}")
        else:
            self.logger.info("No valid best trial config found; skipping save.")

        return best_trial.config if best_trial else {}

def get_activation(activation: str):
    """Maps a string to a PyTorch activation module."""
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def generate_hidden_dims(init_dim: int, min_dim: int = 2) -> List[int]:
    """
    Generates a list of hidden dimensions by halving the initial dimension until the minimum is reached.
    For example, if init_dim is 64, this returns [64, 32, 16, 8, 4, 2].
    """
    dims = []
    current = init_dim
    while current >= min_dim:
        dims.append(current)
        current = current // 2
    return dims

class AutoEncoder(nn.Module):
    """
    Generic Autoencoder for configurable reduction.
    Builds encoder and decoder layers based on a list of hidden dimensions.
    Allows tuning of dropout, activation, and network architecture.
    """
    def __init__(self, input_dim: int, hidden_dims: Union[int, Sequence[int]] = 64, compressed_dim: int = 1,dropout: float = 0.0, activation: str = "relu"):
        super(AutoEncoder, self).__init__()
        self.activation = get_activation(activation)

        if isinstance(hidden_dims, (int, float)):
            hidden_dims = generate_hidden_dims(int(hidden_dims))
        elif not isinstance(hidden_dims, list):
            raise ValueError("hidden_dims must be an int, float, or a list of ints.")

        # encoder:
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, compressed_dim))
        # the * operator unpacks the list into separate arguments
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder:
        decoder_layers = []
        current_dim = compressed_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
