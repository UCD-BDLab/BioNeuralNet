"""
Download TCGA data directly using the GDC API and organize it for BioNeuralNet.

This script downloads gene expression, clinical, and other omics data
from TCGA and organizes it in the BioNeuralNet dataset structure.
"""

import os
import requests # type: ignore
import json
import pandas as pd # type: ignore
import tarfile
import gzip
import shutil
from pathlib import Path
import time
from tqdm import tqdm

GDC_API_BASE = "https://api.gdc.cancer.gov/"
GDC_FILES_ENDPOINT = GDC_API_BASE + "files"
GDC_CASES_ENDPOINT = GDC_API_BASE + "cases"
GDC_DATA_ENDPOINT = GDC_API_BASE + "data"

OUTPUT_BASE = Path(__file__).resolve().parents[1] / "bioneuralnet" / "datasets"


class TCGADownloader:
    """Download and organize TCGA data for BioNeuralNet."""

    def __init__(self, cancer_type="BRCA", output_dir=None):
        """
        Initialize the downloader.

        Args:
            cancer_type: TCGA cancer type (e.g., "BRCA" for breast cancer)
            output_dir: Output directory (default: bioneuralnet/datasets/{cancer_type})
        """
        self.cancer_type = cancer_type
        if output_dir is None:
            self.output_dir = OUTPUT_BASE / cancer_type.lower()
        else:
            self.output_dir = Path(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        # Initializing data containers
        self.gene_expression_files = []
        self.clinical_files = []
        self.methylation_files = []
        self.mutation_files = []
        self.case_ids = []

    def search_files(self, data_category=None, data_type=None, workflow_type=None,
                    experimental_strategy=None, access="open", format=None):
        """
        Search for files in GDC.

        Args:
            data_category: Data category (e.g., "Transcriptome Profiling")
            data_type: Data type (e.g., "Gene Expression Quantification")
            workflow_type: Workflow type (e.g., "HTSeq - FPKM")
            experimental_strategy: Experimental strategy (e.g., "RNA-Seq")
            access: Access type (e.g., "open" or "controlled")
            format: File format (e.g., "TSV")

        Returns:
            List of file IDs
        """
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [f"TCGA-{self.cancer_type}"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "access",
                        "value": [access]
                    }
                }
            ]
        }

        if data_category:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": [data_category]
                }
            })

        if data_type:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type]
                }
            })

        if workflow_type:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": [workflow_type]
                }
            })

        if experimental_strategy:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": [experimental_strategy]
                }
            })

        if format:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "data_format",
                    "value": [format]
                }
            })

        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,cases.case_id,data_type,data_format",
            "format": "JSON",
            "size": "1000"
        }

        print(f"Searching with filters: {json.dumps(filters, indent=2)}")

        response = requests.get(GDC_FILES_ENDPOINT, params=params)

        if response.status_code == 200:
            files = response.json().get("data", {}).get("hits", [])
            print(f"Found {len(files)} files matching criteria")
            return files
        else:
            print(f"Error searching files: {response.status_code}")
            print(f"Response: {response.text}")
            return []

    def download_files(self, file_ids, output_dir):
        """
        Download files from GDC.

        Args:
            file_ids: List of file IDs
            output_dir: Output directory

        Returns:
            List of downloaded file paths
        """
        if not file_ids:
            print("No files to download")
            return []

        os.makedirs(output_dir, exist_ok=True)

        # Limiting to 5 files to avoid connection issues
        file_ids = file_ids[:5]
        print(f"Downloading {len(file_ids)} files...")

        data = {"ids": file_ids}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(GDC_DATA_ENDPOINT,
                                       data=json.dumps(data),
                                       headers={"Content-Type": "application/json"},
                                       stream=True,
                                       timeout=300)

                if response.status_code == 200:
                    content_disp = response.headers.get("Content-Disposition")
                    if content_disp:
                        filename = content_disp.split("filename=")[1].strip("\"")
                    else:
                        filename = "gdc_download.tar.gz"

                    tarball_path = os.path.join(output_dir, filename)
                    with open(tarball_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)

                    print(f"Downloaded tarball: {tarball_path}")

                    try:
                        with tarfile.open(tarball_path) as tar:
                            tar.extractall(path=output_dir)

                        os.remove(tarball_path)

                        extracted_files = []
                        for root, dirs, files in os.walk(output_dir):
                            for file in files:
                                if file.endswith(".gz") or file.endswith(".tsv") or file.endswith(".txt"):
                                    extracted_files.append(os.path.join(root, file))

                        print(f"Extracted {len(extracted_files)} files")
                        return extracted_files
                    except Exception as e:
                        print(f"Error extracting files: {e}")
                        return []
                else:
                    print(f"Error downloading files: {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(5)
                        continue
                    else:
                        return []

            except requests.exceptions.RequestException as e:
                print(f"Connection error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(5)
                    continue
                else:
                    print("Max retries reached, giving up")
                    return []

        return []

    def process_gene_expression(self, files):
        """
        Process gene expression files.

        Args:
            files: List of gene expression file paths (can be directories or files)

        Returns:
            DataFrame with gene expression data
        """
        print("Processing gene expression files...")

        gene_expr_df = None

        tsv_files = []
        for file_path in files:
            if os.path.isdir(file_path):
                for root, dirs, files_in_dir in os.walk(file_path):
                    for file in files_in_dir:
                        if file.endswith('.tsv') and 'gene_counts' in file:
                            tsv_files.append(os.path.join(root, file))
            elif file_path.endswith('.tsv') or file_path.endswith('.gz'):
                tsv_files.append(file_path)

        print(f"Found {len(tsv_files)} gene expression TSV files to process")

        for file_path in tqdm(tsv_files):
            try:
                file_name = os.path.basename(file_path)
                sample_id = os.path.basename(os.path.dirname(file_path))
                if not sample_id or sample_id == 'gene_expression':
                    sample_id = file_name.split(".")[0]

                if file_path.endswith(".gz"):
                    with gzip.open(file_path, "rt") as f:
                        df = pd.read_csv(f, sep="\t", comment='#')
                else:
                    df = pd.read_csv(file_path, sep="\t", comment='#')

                if 'gene_id' in df.columns and 'unstranded' in df.columns:
                    df = df[['gene_id', 'gene_name', 'unstranded']].copy()
                    df.columns = ['gene_id', 'gene_name', 'count']
                elif 'gene_id' in df.columns and len(df.columns) >= 2:
                    count_col = df.columns[1]
                    df = df[['gene_id', count_col]].copy()
                    df.columns = ['gene_id', 'count']
                    df['gene_name'] = df['gene_id'].apply(lambda x: str(x).split(".")[0])
                else:
                    print(f"Unknown file format for {file_path}, skipping...")
                    continue

                df["count"] = pd.to_numeric(df["count"], errors="coerce")
                df = df.dropna(subset=["count"])

                df = df.dropna(subset=["gene_name"])

                sample_df = df.groupby("gene_name")["count"].mean().to_frame()
                sample_df.columns = [sample_id]

                if gene_expr_df is None:
                    gene_expr_df = sample_df
                else:
                    gene_expr_df = gene_expr_df.join(sample_df, how="outer")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        if gene_expr_df is not None:
            gene_expr_df = gene_expr_df.fillna(0)
            print(f"Processed gene expression data: {gene_expr_df.shape[0]} genes x {gene_expr_df.shape[1]} samples")

        return gene_expr_df

    def process_clinical(self, files):
        """
        Process clinical files.

        Args:
            files: List of clinical file paths (can be directories or files)

        Returns:
            DataFrame with clinical data
        """
        print("Processing clinical files...")

        clinical_df = None

        clinical_files = []
        for file_path in files:
            if os.path.isdir(file_path):
                for root, dirs, files_in_dir in os.walk(file_path):
                    for file in files_in_dir:
                        if file.endswith('.xml') or file.endswith('.tsv') or file.endswith('.txt'):
                            clinical_files.append(os.path.join(root, file))
            elif file_path.endswith(('.xml', '.tsv', '.txt', '.gz')):
                clinical_files.append(file_path)

        print(f"Found {len(clinical_files)} clinical files to process")

        sample_ids = []

        for file_path in tqdm(clinical_files):
            try:
                file_name = os.path.basename(file_path)

                if file_path.endswith('.xml'):
                    sample_id = os.path.basename(os.path.dirname(file_path))
                    if sample_id and sample_id != 'clinical':
                        sample_ids.append(sample_id)
                elif file_path.endswith(('.tsv', '.txt')):
                    try:
                        if file_path.endswith('.gz'):
                            with gzip.open(file_path, 'rt') as f:
                                df = pd.read_csv(f, sep='\t', nrows=100)  # Read first 100 rows
                        else:
                            df = pd.read_csv(file_path, sep='\t', nrows=100)

                        for col in df.columns:
                            if 'sample' in col.lower() or 'barcode' in col.lower():
                                sample_ids.extend(df[col].dropna().astype(str).tolist())
                                break
                    except:
                        pass

            except Exception as e:
                print(f"Error processing clinical file {file_path}: {e}")
                continue

        if sample_ids:
            unique_samples = list(set(sample_ids))
            clinical_df = pd.DataFrame({
                'sample_id': unique_samples,
                'dataset': ['TCGA-BRCA'] * len(unique_samples),
                'data_type': ['clinical'] * len(unique_samples)
            })
            clinical_df.set_index('sample_id', inplace=True)
            print(f"Processed clinical data: {len(unique_samples)} samples")
        else:
            print("No clinical sample information found")

        return clinical_df

    def download_gene_expression(self):
        """Download gene expression data."""
        print("Searching for gene expression files...")

        # First, trying to find any RNA-Seq gene expression files without specific workflow
        files = self.search_files(
            data_category="Transcriptome Profiling",
            data_type="Gene Expression Quantification",
            experimental_strategy="RNA-Seq"
        )

        if not files:
            print("No basic gene expression files found, trying STAR - Counts...")
            files = self.search_files(
                data_category="Transcriptome Profiling",
                data_type="Gene Expression Quantification",
                workflow_type="STAR - Counts",
                experimental_strategy="RNA-Seq"
            )

        if not files:
            print("No STAR files found, trying HTSeq-FPKM...")
            files = self.search_files(
                data_category="Transcriptome Profiling",
                data_type="Gene Expression Quantification",
                workflow_type="HTSeq - FPKM",
                experimental_strategy="RNA-Seq"
            )

        if not files:
            print("No HTSeq-FPKM files found, trying HTSeq-Counts...")
            files = self.search_files(
                data_category="Transcriptome Profiling",
                data_type="Gene Expression Quantification",
                workflow_type="HTSeq - Counts",
                experimental_strategy="RNA-Seq"
            )

        if not files:
            print("Trying without workflow restriction...")
            files = self.search_files(
                data_category="Transcriptome Profiling",
                data_type="Gene Expression Quantification"
            )

        if not files:
            print("No gene expression files found")
            return

        print(f"Found {len(files)} gene expression files")

        file_ids = [file["file_id"] for file in files]

        output_dir = self.output_dir / "raw" / "gene_expression"
        downloaded_files = self.download_files(file_ids[:10], output_dir)  # Limit to 10 files for testing

        if not downloaded_files:
            print("No gene expression files downloaded")
            return

        print(f"Downloaded {len(downloaded_files)} gene expression files")

        gene_expr_df = self.process_gene_expression(downloaded_files)

        gene_expr_path = self.output_dir / "gene_expression.csv"
        gene_expr_df.to_csv(gene_expr_path)

        print(f"Gene expression data saved to {gene_expr_path}")

    def download_clinical(self):
        """Download clinical data."""
        print("Searching for clinical files...")

        files = self.search_files(
            data_category="Clinical",
            data_type="Clinical Supplement"
        )

        if not files:
            print("No Clinical Supplement files found, trying Clinical Data...")
            files = self.search_files(
                data_category="Clinical",
                data_type="Clinical Data"
            )

        if not files:
            print("Trying any clinical files...")
            files = self.search_files(
                data_category="Clinical"
            )

        if not files:
            print("No clinical files found")
            return

        print(f"Found {len(files)} clinical files")

        file_ids = [file["file_id"] for file in files]

        output_dir = self.output_dir / "raw" / "clinical"
        downloaded_files = self.download_files(file_ids[:10], output_dir)  # Limit to 10 files for testing

        if not downloaded_files:
            print("No clinical files downloaded")
            return

        print(f"Downloaded {len(downloaded_files)} clinical files")

        clinical_df = self.process_clinical(downloaded_files)

        clinical_path = self.output_dir / "clinical.csv"
        if clinical_df is None or (hasattr(clinical_df, "empty") and clinical_df.empty):
            print("No clinical sample information found; skipping clinical.csv save.")
        else:
            clinical_df.to_csv(clinical_path, index=False)
            print(f"Clinical data saved to {clinical_path}")

    def download_all(self):
        """Download all data types."""
        print(f"Downloading TCGA-{self.cancer_type} data...")

        os.makedirs(self.output_dir, exist_ok=True)

        self.download_gene_expression()

        self.download_clinical()

        print("Download complete")


def download_string_ppi(output_dir=None):
    """
    Download STRING PPI network.

    Args:
        output_dir: Output directory (default: bioneuralnet/datasets/string)
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE / "string"

    os.makedirs(output_dir, exist_ok=True)

    # URL for STRING PPI network (human, score > 700)
    string_url = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"

    # Downloading the file
    print(f"Downloading STRING PPI network from {string_url}...")
    response = requests.get(string_url, stream=True)

    if response.status_code == 200:
        file_path = output_dir / "string_ppi.txt.gz"
        with open(file_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)

        print(f"STRING PPI network saved to {file_path}")

        # Downloading STRING mapping
        mapping_url = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"
        print(f"Downloading STRING mapping from {mapping_url}...")
        response = requests.get(mapping_url, stream=True)

        if response.status_code == 200:
            mapping_path = output_dir / "string_mapping.txt.gz"
            with open(mapping_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024)):
                    if chunk:
                        f.write(chunk)

            print(f"STRING mapping saved to {mapping_path}")
        else:
            print(f"Error downloading STRING mapping: {response.status_code}")
    else:
        print(f"Error downloading STRING PPI network: {response.status_code}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TCGA and STRING data for BioNeuralNet")
    parser.add_argument("--cancer-type", type=str, default="BRCA", help="TCGA cancer type (default: BRCA)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--string", action="store_true", help="Download STRING PPI network")
    parser.add_argument("--tcga", action="store_true", help="Download TCGA data")

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_BASE / args.cancer_type.lower()

    if args.string:
        download_string_ppi(output_dir / "string")

    if args.tcga:
        downloader = TCGADownloader(args.cancer_type, output_dir)
        downloader.download_all()

    if not args.string and not args.tcga:
        # Default: download both
        download_string_ppi(output_dir / "string")
        downloader = TCGADownloader(args.cancer_type, output_dir)
        downloader.download_all()
