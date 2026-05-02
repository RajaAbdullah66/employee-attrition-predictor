"""
src/data_loader.py
------------------
Responsibility: Load the raw CSV dataset and return a clean DataFrame.

What this file does:
  1. Reads the IBM HR Attrition CSV file from the data/ folder.
  2. Drops columns that are completely useless for prediction.
  3. Converts the target column 'Attrition' from Yes/No text to 1/0 numbers.
  4. Prints a summary so you can see the data before training.

Why drop EmployeeCount, Over18, StandardHours?
  These columns have the SAME value for every single employee
  (EmployeeCount=1, Over18=Y, StandardHours=80).
  A column with no variation teaches the model nothing.
  Keeping them wastes memory and can confuse some models.

Why convert Attrition to 0/1?
  All ML models work with numbers, not text.
  Yes → 1 means "employee left"
  No  → 0 means "employee stayed"
"""

import os
import pandas as pd

# Path to the dataset file
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Columns that have the same value for everyone → useless for prediction
USELESS_COLUMNS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]


def load_raw_data() -> pd.DataFrame:
    """
    Load the CSV and return a clean DataFrame.

    Returns
    -------
    df : pandas DataFrame with 1470 rows and ~31 useful columns.
         Target column 'Attrition' is 1 (left) or 0 (stayed).
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please download 'WA_Fn-UseC_-HR-Employee-Attrition.csv' from Kaggle\n"
            "and place it inside the data/ folder."
        )

    # Load CSV into a pandas DataFrame
    df = pd.read_csv(DATA_PATH)
    print(f"[data_loader] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Drop columns that carry no predictive information
    cols_to_drop = [c for c in USELESS_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[data_loader] Dropped useless columns: {cols_to_drop}")

    # Convert target column: Yes → 1, No → 0
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Show class distribution
    left   = df["Attrition"].sum()
    stayed = len(df) - left
    print(f"[data_loader] Attrition: {left} left ({left/len(df)*100:.1f}%)  |  "
          f"{stayed} stayed ({stayed/len(df)*100:.1f}%)")
    print(f"[data_loader] Class imbalance ratio: 1 : {stayed/left:.1f}")

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple:
    """
    Separate features (X) from target (y).

    Returns
    -------
    X : DataFrame of all input features
    y : Series of 0/1 labels
    """
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    return X, y


def get_column_types(df: pd.DataFrame) -> tuple:
    """
    Identify which columns are categorical (text) vs numerical (numbers).
    This is used by feature_engineering.py to know what to encode vs scale.

    Returns
    -------
    categorical_cols : list of column names with text values
    numerical_cols   : list of column names with number values
    """
    X = df.drop(columns=["Attrition"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"[data_loader] Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"[data_loader] Numerical  columns ({len(numerical_cols)}): {numerical_cols[:5]}...")

    return categorical_cols, numerical_cols


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head(3))
    print(df.dtypes)
