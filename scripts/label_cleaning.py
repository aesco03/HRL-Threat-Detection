"""
clean_pdfmalware2022.py
-----------------------
Cleans and preprocesses the CIC-PDFMalware2022 dataset (.parquet file)
to fix inconsistent datatypes, unexpected values, and categorical mislabels.

Author: Adrian Escobar (with GPT-5)
"""

import pandas as pd
import numpy as np
import re
import os

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
INPUT_PATH = "data/raw/pdfs/malware/PDFMalware2022.parquet" 
OUTPUT_PATH = "data/processed/labeled/cleaned_pdfmalware2022.parquet"

# -----------------------------------------------
# PREPARE OUTPUT DIRECTORY
# -----------------------------------------------
# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -----------------------------------------------
# EXPECTED FEATURE TYPES
# -----------------------------------------------
NUMERIC_LIKE_COLS = [
    'Images', 'Obj', 'Endobj', 'Endstream', 'Xref',
    'StartXref', 'PageNo', 'JS', 'Javascript'
]

TEXT_LIKE_COLS = ['FileName', 'Title_Characters', 'Text']

# -----------------------------------------------
# CLEANING UTILITIES
# -----------------------------------------------
def clean_numeric_value(val):
    """Convert dataset entries to numeric where possible."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val

    val = str(val).strip()
    if val.lower() in ["pdfid.py", "bytes[endheader]", "list", "_pro_rodeo_pix_", "most"]:
        return np.nan

    # Extract numeric part (e.g. '2(2)' → 2)
    match = re.match(r"^(-?\d+)", val)
    if match:
        return int(match.group(1))

    return np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Main cleaning function with class-specific imputation."""
    df = df.copy()

    # ... (All previous steps like column normalization, type conversion, 
    # dropping text columns, standardizing 'Class', and replacing negatives with NaN remain here)
    
    # Standardize label column (ensure this runs before imputation)
    # Standardize label column
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(str).str.strip().str.lower()
        # Add the 'inplace=False' and infer_objects to explicitly handle downcasting
        df['Class'] = df['Class'].replace({'malicious': 1, 'benign': 0, '0': 0, '1': 1}).infer_objects(copy=False)
    # Replace impossible negatives with NaN (ensure this runs before imputation)
    for col in NUMERIC_LIKE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_value)
            # Ensure the column is explicitly converted to a float type
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    # Drop constant or empty columns
    nunique = df.nunique()
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Drop rows that are completely NaN
    df.dropna(axis=0, how='all', inplace=True)

    # -------------------------------------------------------------------
    # MODIFIED STEP: Fill missing numeric values with CLASS-SPECIFIC MEDIANS
    # -------------------------------------------------------------------
    if 'Class' in df.columns:
        print("Imputing missing values with class-specific medians...")
        for col in NUMERIC_LIKE_COLS:
            if col in df.columns:
                # Calculate the median for each class (Malicious and Benign)
                class_medians = df.groupby('Class')[col].median()
                
                # Use a transform to fill NaN values based on the group's median
                # This fills NaNs in Benign rows with the Benign median, and NaNs 
                # in Malicious rows with the Malicious median.
                df[col] = df.groupby('Class')[col].transform(lambda x: x.fillna(x.median()))
    else:
        # Fallback to global median if the 'Class' column is missing for some reason
        print("Warning: 'Class' column missing. Falling back to global median imputation.")
        for col in NUMERIC_LIKE_COLS:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
                
    # -------------------------------------------------------------------

    print(f"✅ Cleaned dataset shape: {df.shape}")
    return df


# -----------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    print("Cleaning dataset...")
    cleaned_df = clean_dataframe(df)

    print(f" Saving cleaned file to {OUTPUT_PATH} ...")
    cleaned_df.to_parquet(OUTPUT_PATH, index=False)

    print("Cleaning complete! File ready for ML pipelines.")
