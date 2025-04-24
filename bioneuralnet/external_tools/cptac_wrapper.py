
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# try:
#     import cptac

# except ImportError:
#     raise ImportError("Please install external module: pip install Node2Vec")

# def get_cancer_data(cancer_class):
#     """Load cancer_type dataset and return genomics, proteomics, and clinical data."""
    
#     cancer_type = cancer_class
    
#     genomics = cancer_type.get_CNV(source="bcm")
#     proteomics = cancer_type.get_proteomics(source="umich")
#     clinical = cancer_type.get_clinical(source="mssm")
    
#     proteomics = cptac.utils.reduce_multiindex(proteomics, levels_to_drop="Name", quiet=True)
#     genomics = cptac.utils.reduce_multiindex(genomics, levels_to_drop="Name", quiet=True)
    
#     genomics.columns.name = None
#     proteomics.columns.name = None
#     clinical.columns.name = None

#     return genomics, proteomics, clinical

# def preprocess_clinical(clinical, selected_columns=None, target_map="tumor_stage_pathological"):
#     """Preprocess clinical data, encoding categorical values and handling missing values."""

#     if selected_columns is None:
#         selected_columns = [
#             "tumor_stage_pathological", "age", "sex", "tumor_site", "tumor_laterality",
#             "tumor_focality", "tumor_size_cm", "histologic_grade", "bmi",
#             "alcohol_consumption", "tobacco_smoking_history", "medical_condition"
#         ]

#     df = clinical[selected_columns].copy()
    
#     mapping = {"Stage I": 0, "Stage II": 1, "Stage III": 1, "Stage IV": 1}
#     df["tumor_stage_pathological"] = df["tumor_stage_pathological"].map(mapping)

#     phenotype = df["tumor_stage_pathological"].fillna(0).astype(int)
#     phenotype_df = phenotype.to_frame(name="tumor_stage_pathological")
    
#     df.drop(columns=["tumor_stage_pathological"], inplace=True)
    
#     categorical_cols = [
#         "sex", "tumor_site", "tumor_laterality", "tumor_focality", "histologic_grade",
#         "alcohol_consumption", "tobacco_smoking_history", "medical_condition"
#     ]
    
#     numerical_cols = list(set(selected_columns) - set(categorical_cols))

#     # remove tartget column from numerical_cols
#     numerical_cols.remove(target_map)
    
#     encoder = LabelEncoder()
#     for col in categorical_cols:
#         df[col] = encoder.fit_transform(df[col].astype(str).fillna("Missing"))
    
#     for col in numerical_cols:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#         df[col] = df[col].fillna(df[col].median())
    
#     return df, phenotype_df

# def filter_common_patients(genomics, proteomics, clinical):
#     """Keep only common patients across all datasets."""
#     common_patients = clinical.index.intersection(genomics.index).intersection(proteomics.index)
#     print(f"Common Patients: {len(common_patients)}")
    
#     return genomics.loc[common_patients], proteomics.loc[common_patients], clinical.loc[common_patients]
