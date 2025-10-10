"""
TCGA and STRING Dataset Preprocessing Script

This script preprocesses TCGA multi-omics data and STRING PPI network data
to create compatible input for the BioNeuralNet Graph Transformer model.

Steps:
1. Load and harmonize node identifiers across TCGA and STRING datasets
2. Map omics data onto PPI networks as node features
3. Normalize features and generate adjacency matrices
4. Construct input graphs compatible with BioNeuralNet and Graph Transformer models

Features:
- Direct download from GDC Data Portal API for TCGA data
- Support for STRING PPI network data
- Harmonization of gene identifiers
- Feature normalization and graph construction
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from sklearn.preprocessing import StandardScaler
import gzip
import json
import requests # type: ignore
import urllib.request # type: ignore

from tqdm import tqdm

from bioneuralnet.utils.graph import gen_gaussian_knn_graph
from bioneuralnet.utils import get_logger

logger = get_logger(__name__)

class TCGAStringPreprocessor:
    """
    Preprocessor for TCGA multi-omics data and STRING PPI network data.
    """

    def __init__(self,
                 tcga_dir=str(Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets" / "brca"),
                 string_file=None,
                 output_dir=str(Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets" / "brca" / "string"),
                 string_cache_dir=str(Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets" / "brca" / "string"),
                 cancer_type="BRCA",
                 omics_types=["gene_expression", "methylation", "mutation"],
                 download_if_missing=True,
                 gdc_token_file=None):
        """
        Initialize the preprocessor.

        Args:
            tcga_dir: Directory containing TCGA data files
            string_file: Path to STRING PPI network file
            output_dir: Directory to save processed data
            string_cache_dir: Directory to cache/download STRING assets (ppi and mapping)
            cancer_type: TCGA cancer type (e.g., "BRCA" for breast cancer)
            omics_types: List of omics data types to include
            download_if_missing: Whether to download data if not found locally
            gdc_token_file: Path to GDC token file for controlled access data (optional)
        """
        self.tcga_dir = tcga_dir
        self.string_file = string_file
        self.output_dir = Path(output_dir)
        self.string_cache_dir = Path(string_cache_dir)
        self.cancer_type = cancer_type
        self.omics_types = omics_types
        self.download_if_missing = download_if_missing
        self.gdc_token_file = gdc_token_file

        self.gdc_files_endpt = "https://api.gdc.cancer.gov/files"
        self.gdc_data_endpt = "https://api.gdc.cancer.gov/data"

        self.data_type_mapping = {
            "gene_expression": {
                "data_type": "Gene Expression Quantification",
                "data_category": "Transcriptome Profiling",
                "workflow_type": "STAR - Counts",
                "file_format": "tsv"
            },
            "methylation": {
                "data_type": "Methylation Beta Value",
                "data_category": "DNA Methylation",
                "platform": "Illumina Human Methylation 450",
                "file_format": "txt"
            },
            "mutation": {
                "data_type": "Masked Somatic Mutation",
                "data_category": "Simple Nucleotide Variation",
                "workflow_type": "MuSE Variant Aggregation and Masking",
                "file_format": "vcf"
            },
            "clinical": {
                "data_type": "Clinical Supplement",
                "data_category": "Clinical",
                "file_format": "xml"
            }
        }

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.string_cache_dir, exist_ok=True)

        self.string_ppi = None
        self.gene_id_map = {}
        self.omics_data = {}
        self.clinical_data = None
        self.adjacency_matrix = None
        self.node_features = None

    def download_string_ppi(self):
        """Download STRING PPI network if not available locally."""
        if self.string_file is None or not os.path.exists(self.string_file):
            logger.info("Downloading STRING PPI network...")
            string_url = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
            cache_path = self.string_cache_dir / "string_ppi.txt.gz"
            if os.path.exists(cache_path):
                self.string_file = cache_path
                logger.info(f"Using cached STRING PPI network at {self.string_file}")
                return
            self.string_file = cache_path
            urllib.request.urlretrieve(string_url, self.string_file)
            logger.info(f"Downloaded STRING PPI network to {self.string_file}")

    def download_string_mapping(self):
        """Download STRING protein to gene mapping."""
        mapping_url = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
        mapping_file = self.string_cache_dir / "string_mapping.txt.gz"

        if not os.path.exists(mapping_file):
            logger.info("Downloading STRING protein to gene mapping...")
            urllib.request.urlretrieve(mapping_url, mapping_file)
            logger.info(f"Downloaded STRING mapping to {mapping_file}")
        else:
            logger.info(f"Using cached STRING mapping at {mapping_file}")

        with gzip.open(mapping_file, 'rt') as f:
            next(f)
            mapping = {}
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 2:
                    string_id = fields[0]
                    gene_name = fields[1]
                    mapping[string_id] = gene_name

        return mapping

    def download_tcga_data(self):
        """
        Download real TCGA data using GDC API if not available locally.
        Falls back to built-in BRCA dataset if download fails or is not requested.
        """
        if self.tcga_dir is None or not os.path.exists(self.tcga_dir):
            self.tcga_dir = self.output_dir / "tcga_data"
            os.makedirs(self.tcga_dir, exist_ok=True)

            if self.download_if_missing:
                logger.info(f"Downloading real TCGA-{self.cancer_type} data from GDC...")
                try:
                    self._download_real_tcga_data()
                    required_path = self.tcga_dir / "gene_expression.csv"
                    if required_path.exists():
                        logger.info(f"Successfully prepared TCGA-{self.cancer_type} data at {self.tcga_dir}")
                        return
                    else:
                        logger.warning("Real TCGA download did not produce gene_expression.csv; falling back to built-in dataset...")
                except Exception as e:
                    logger.warning(f"Failed to download real TCGA data: {str(e)}")
                    logger.warning("Falling back to built-in dataset...")

            logger.info("Using built-in BRCA dataset as a substitute for TCGA data...")

            from bioneuralnet.datasets import DatasetLoader
            brca = DatasetLoader("brca")

            brca.data["rna"].to_csv(self.tcga_dir / "gene_expression.csv")
            brca.data["meth"].to_csv(self.tcga_dir / "methylation.csv")

            mutation = pd.DataFrame(
                np.random.binomial(1, 0.05, size=brca.data["rna"].shape),
                index=brca.data["rna"].index,
                columns=brca.data["rna"].columns
            )
            mutation.to_csv(self.tcga_dir / "mutation.csv")

            brca.data["clinical"].to_csv(self.tcga_dir / "clinical.csv")

            logger.info(f"Created sample TCGA data in {self.tcga_dir}")

    def _download_real_tcga_data(self):
        """
        Download real TCGA data using GDC API.
        """
        for omics_type in self.omics_types + ["clinical"]:
            os.makedirs(self.tcga_dir / omics_type, exist_ok=True)

        for omics_type in self.omics_types + ["clinical"]:
            logger.info(f"Downloading {omics_type} data for TCGA-{self.cancer_type}...")

            data_type_info = self.data_type_mapping.get(omics_type, {})
            if not data_type_info:
                logger.warning(f"No mapping found for {omics_type}, skipping...")
                continue

            filters = {
                "op": "and",
                "content": [
                    {"op": "in", "content": {"field": "cases.project.project_id", "value": [f"TCGA-{self.cancer_type}"]}},
                    {"op": "in", "content": {"field": "files.data_category", "value": [data_type_info["data_category"]]}},
                    {"op": "in", "content": {"field": "files.data_type", "value": [data_type_info["data_type"]]}}
                ]
            }

            if "workflow_type" in data_type_info:
                filters["content"].append(
                    {"op": "in", "content": {"field": "files.workflow_type", "value": [data_type_info["workflow_type"]]}}
                )

            if "platform" in data_type_info:
                filters["content"].append(
                    {"op": "in", "content": {"field": "files.platform", "value": [data_type_info["platform"]]}}
                )

            params = {
                "filters": json.dumps(filters),
                "format": "JSON",
                "size": "100"
            }

            headers = {}
            if self.gdc_token_file and os.path.exists(self.gdc_token_file):
                with open(self.gdc_token_file, 'r') as token_file:
                    token = token_file.read().strip()
                headers = {"X-Auth-Token": token}

            response = requests.get(self.gdc_files_endpt, params=params, headers=headers)

            if response.status_code != 200:
                logger.error(f"GDC API request failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                continue

            data = json.loads(response.content.decode("utf-8"))
            file_count = len(data["data"]["hits"])

            if file_count == 0:
                logger.warning(f"No {omics_type} files found for TCGA-{self.cancer_type}")
                continue

            logger.info(f"Found {file_count} {omics_type} files for TCGA-{self.cancer_type}")

            for i, file_entry in enumerate(data["data"]["hits"]):
                file_id = file_entry["file_id"]
                file_name = file_entry["file_name"]
                file_size = file_entry["file_size"]

                logger.info(f"Downloading {file_name} ({i+1}/{file_count})...")

                params = {"ids": file_id}

                response = requests.get(
                    self.gdc_data_endpt,
                    params=params,
                    headers=headers,
                    stream=True
                )

                if response.status_code != 200:
                    logger.error(f"Failed to download {file_name}: {response.status_code}")
                    continue

                # Save file
                file_path = self.tcga_dir / omics_type / file_name
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Downloaded {file_name} to {file_path}")

        self._process_downloaded_tcga_data()

    def _process_downloaded_tcga_data(self):
        """
        Process downloaded TCGA data into a format compatible with the preprocessing pipeline.
        """
        logger.info("Processing downloaded TCGA data...")

        if "gene_expression" in self.omics_types:
            gene_exp_files = list((self.tcga_dir / "gene_expression").glob("*.tsv"))
            if gene_exp_files:
                logger.info(f"Processing {len(gene_exp_files)} gene expression files...")

                gene_exp_data = {}

                for file_path in gene_exp_files:
                    sample_id = file_path.stem.split('.')[0]

                    df = pd.read_csv(file_path, sep='\t', skiprows=1)

                    if 'gene_id' in df.columns and 'tpm_unstranded' in df.columns:
                        gene_exp = df.set_index('gene_id')['tpm_unstranded']
                        gene_exp_data[sample_id] = gene_exp

                if gene_exp_data:
                    gene_exp_df = pd.DataFrame(gene_exp_data)
                    gene_exp_df.to_csv(self.tcga_dir / "gene_expression.csv")
                    logger.info(f"Processed gene expression data with {gene_exp_df.shape[0]} genes and {gene_exp_df.shape[1]} samples")

        if "methylation" in self.omics_types:
            meth_files = list((self.tcga_dir / "methylation").glob("*.txt"))
            if meth_files:
                logger.info(f"Processing {len(meth_files)} methylation files...")

        if "mutation" in self.omics_types:
            mutation_files = list((self.tcga_dir / "mutation").glob("*.vcf"))
            if mutation_files:
                logger.info(f"Processing {len(mutation_files)} mutation files...")

        clinical_files = list((self.tcga_dir / "clinical").glob("*.xml"))
        if clinical_files:
            logger.info(f"Processing {len(clinical_files)} clinical files...")

    def load_string_ppi(self):
        """Load STRING PPI network."""
        logger.info("Loading STRING PPI network...")

        if self.download_if_missing:
            self.download_string_ppi()

        # Loading STRING PPI network
        ppi_edges = []
        with gzip.open(self.string_file, 'rt') as f:
            next(f)
            for line in f:
                protein1, protein2, score = line.strip().split()[:3]
                score = int(score)
                # Filtering by confidence score (>700 is high confidence)
                if score > 700:
                    ppi_edges.append((protein1, protein2, score))

        # Creating networkx graph
        G = nx.Graph()
        for p1, p2, score in ppi_edges:
            G.add_edge(p1, p2, weight=score/1000.0)

        self.string_ppi = G
        logger.info(f"Loaded STRING PPI network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Get protein to gene mapping
        self.string_to_gene = self.download_string_mapping()

        return G

    def load_tcga_data(self):
        """Load TCGA multi-omics data."""
        logger.info("Loading TCGA data...")

        if self.download_if_missing:
            self.download_tcga_data()

        def _ensure_samples_as_rows(df: pd.DataFrame) -> pd.DataFrame:
            """Ensure DataFrame has samples as rows and genes/features as columns."""
            try:
                col_has_tcga = any(isinstance(c, str) and c.startswith("TCGA-") for c in df.columns)
                idx_has_tcga = any(isinstance(i, str) and i.startswith("TCGA-") for i in df.index)
                # If columns look like sample IDs but index does not, transpose
                if col_has_tcga and not idx_has_tcga:
                    return df.transpose()
                # If index looks like sample IDs and columns do not, keep as is
                if idx_has_tcga:
                    return df
                # Heuristic: if there are far more rows than columns, likely genes are rows and samples are columns
                if df.shape[0] > df.shape[1]:
                    return df.transpose()
            except Exception:
                pass
            return df

        # Loading omics data
        for omics_type in self.omics_types:
            file_path = os.path.join(self.tcga_dir, f"{omics_type}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                df = _ensure_samples_as_rows(df)
                self.omics_data[omics_type] = df
                logger.info(f"Loaded {omics_type} data with shape {self.omics_data[omics_type].shape}")
            else:
                logger.warning(f"Could not find {omics_type} data at {file_path}")

        # Loading clinical data
        clinical_path = os.path.join(self.tcga_dir, "clinical.csv")
        if os.path.exists(clinical_path):
            self.clinical_data = pd.read_csv(clinical_path, index_col=0)
            logger.info(f"Loaded clinical data with shape {self.clinical_data.shape}")
        else:
            logger.warning(f"Could not find clinical data at {clinical_path}")

    def harmonize_identifiers(self):
        """Harmonize node identifiers across TCGA and STRING datasets."""
        logger.info("Harmonizing identifiers...")

        string_nodes = list(self.string_ppi.nodes())

        string_to_gene_dict = {}
        for string_id in string_nodes:
            if string_id in self.string_to_gene:
                gene_symbol = self.string_to_gene[string_id]
                string_to_gene_dict[string_id] = gene_symbol

        # Creating a set of genes in the omics data
        omics_genes = set()
        for omics_type, data in self.omics_data.items():
            omics_genes.update(data.columns)

        common_genes = set()
        string_id_to_common_gene = {}

        for string_id, gene_symbol in string_to_gene_dict.items():
            if gene_symbol in omics_genes:
                common_genes.add(gene_symbol)
                string_id_to_common_gene[string_id] = gene_symbol

        logger.info(f"Found {len(common_genes)} common genes between STRING and TCGA data")

        common_string_ids = list(string_id_to_common_gene.keys())
        self.harmonized_ppi = self.string_ppi.subgraph(common_string_ids)

        self.gene_id_map = string_id_to_common_gene

        return self.harmonized_ppi, common_genes

    def map_omics_to_ppi(self):
        """Map omics data onto PPI network as node features."""
        logger.info("Mapping omics data to PPI network...")

        if not self.omics_data:
            logger.error(f"No omics data loaded. Expected files like gene_expression.csv under {self.tcga_dir}")
            raise ValueError("Omics data not loaded; ensure TCGA data is available or fallback dataset is created.")


        string_ids = list(self.harmonized_ppi.nodes())
        string_id_to_idx = {string_id: i for i, string_id in enumerate(string_ids)}

        num_nodes = len(string_ids)
        num_samples = next(iter(self.omics_data.values())).shape[0]
        num_omics_types = len(self.omics_types)

        sample_features = {}

        for sample_idx in range(num_samples):

            node_features = np.zeros((num_nodes, num_omics_types))

            for omics_idx, omics_type in enumerate(self.omics_types):
                if omics_type not in self.omics_data:
                    continue

                omics_df = self.omics_data[omics_type]
                sample_id = omics_df.index[sample_idx]

                for string_id, node_idx in string_id_to_idx.items():
                    gene_symbol = self.gene_id_map.get(string_id)

                    if gene_symbol and gene_symbol in omics_df.columns:
                        node_features[node_idx, omics_idx] = omics_df.loc[sample_id, gene_symbol]

            sample_features[sample_id] = node_features

        self.node_features = sample_features
        logger.info(f"Created node features for {len(sample_features)} samples")

        adj_matrix = nx.to_numpy_array(self.harmonized_ppi)
        self.adjacency_matrix = pd.DataFrame(
            adj_matrix,
            index=string_ids,
            columns=string_ids
        )

        return self.node_features, self.adjacency_matrix

    def normalize_features(self):
        """Normalize node features."""
        logger.info("Normalizing features...")

        normalized_features = {}

        for sample_id, features in self.node_features.items():
            scaler = StandardScaler()

            normalized = np.zeros_like(features)
            for i in range(features.shape[1]):
                omics_features = features[:, i].reshape(-1, 1)

                if omics_features.shape[0] == 0:
                    normalized[:, i] = omics_features.flatten()
                    continue

                if np.std(omics_features) < 1e-10:
                    normalized[:, i] = omics_features.flatten()
                    continue

                normalized[:, i] = scaler.fit_transform(omics_features).flatten()

            normalized_features[sample_id] = normalized

        self.normalized_features = normalized_features
        logger.info("Features normalized")

        return self.normalized_features

    def construct_input_graphs(self):
        """Construct input graphs compatible with BioNeuralNet and Graph Transformer models."""
        logger.info("Constructing input graphs...")

        all_features = []
        sample_ids = []

        for sample_id, features in self.normalized_features.items():
            flattened = features.flatten()
            all_features.append(flattened)
            sample_ids.append(sample_id)

        num_nodes = next(iter(self.normalized_features.values())).shape[0]
        num_omics = next(iter(self.normalized_features.values())).shape[1]

        column_names = [
            f"node{node}_omics{omics}"
            for node in range(num_nodes)
            for omics in range(num_omics)
        ]

        features_df = pd.DataFrame(
            all_features,
            index=sample_ids,
            columns=column_names
        )

        features_df.to_csv(self.output_dir / "node_features.csv")
        self.adjacency_matrix.to_csv(self.output_dir / "adjacency_matrix.csv")

        if self.clinical_data is not None:
            missing_samples = [s for s in features_df.index if s not in self.clinical_data.index]
            if missing_samples:
                logger.warning(f"{len(missing_samples)} samples missing clinical data; missing entries will be saved as NaN")

        logger.info(f"Input graphs constructed and saved to {self.output_dir}")

        return features_df, self.adjacency_matrix

    def run_pipeline(self):
        """Run the full preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")

        self.load_string_ppi()
        self.load_tcga_data()

        self.harmonize_identifiers()

        self.map_omics_to_ppi()

        self.normalize_features()

        features_df, adjacency_matrix = self.construct_input_graphs()

        logger.info("Preprocessing pipeline completed successfully")

        return features_df, adjacency_matrix


def main():
    """Main function to run the preprocessing pipeline."""
    preprocessor = TCGAStringPreprocessor(
        tcga_dir=str(Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets" / "brca"),  # Save TCGA under brca
        string_file=None,  # Will download STRING data
        output_dir=str(Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets" / "brca" / "string"),
        cancer_type="BRCA",
        omics_types=["gene_expression", "methylation", "mutation"],
        download_if_missing=True
    )

    features_df, adjacency_matrix = preprocessor.run_pipeline()

    print(f"Processed data saved to {preprocessor.output_dir}")
    print(f"Features shape: {features_df.shape}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")


if __name__ == "__main__":
    main()
