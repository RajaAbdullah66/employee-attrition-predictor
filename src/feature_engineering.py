"""
src/feature_engineering.py
---------------------------
Responsibility: Transform raw features into a format that ML models can use.

Steps performed:
  1. One-Hot Encode categorical columns (Department, JobRole, Gender, etc.)
     → Converts text categories to 0/1 binary columns
  2. Scale numerical columns using StandardScaler
     → Brings all numbers to the same range (mean=0, std=1)
  3. Handle class imbalance using SMOTE
     → Creates synthetic minority (Attrition=1) samples so the model
        does not become biased toward predicting "No Attrition" always

Why One-Hot Encoding?
  ML models cannot do math on text like "Sales" or "Male".
  One-hot encoding creates a separate 0/1 column for each category.
  Example: Department = "Sales" → Dept_Sales=1, Dept_R&D=0, Dept_HR=0

Why StandardScaler?
  Age ranges from 18–60, MonthlyIncome from 1009–19999.
  Without scaling, large-valued features (income) dominate distance-based
  models like KNN and SVM. Scaling fixes this.

Why SMOTE (Synthetic Minority Over-sampling Technique)?
  Only ~23% of employees leave. Without balancing, models learn to always
  predict "Stay" and still get 77% accuracy — but that is useless!
  SMOTE creates NEW artificial "leave" examples by interpolating between
  existing ones. After SMOTE the model sees equal numbers of both classes.
  SMOTE is only applied to training data — NEVER to test data.

The full preprocessor (encoder + scaler) is saved as a joblib file
so the exact same transformations are applied at prediction time.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[feature_engineering] WARNING: imbalanced-learn not installed. "
          "Using class_weight='balanced' instead of SMOTE.")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")


def build_preprocessor(categorical_cols: list, numerical_cols: list):
    """
    Build a sklearn ColumnTransformer that:
      - One-Hot encodes all categorical columns
      - StandardScaler scales all numerical columns

    This is a sklearn Pipeline-compatible object, meaning it can be
    fit on training data and then applied to both train and test data
    consistently.

    Parameters
    ----------
    categorical_cols : list of column names with text data
    numerical_cols   : list of column names with numeric data

    Returns
    -------
    preprocessor : sklearn ColumnTransformer (not yet fitted)
    """
    # One-Hot Encoder: handle_unknown='ignore' means if a new category
    # appears at prediction time that wasn't in training, it becomes all zeros
    # instead of crashing the program
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False    # return dense array, not sparse matrix
    )

    # StandardScaler: subtracts mean and divides by std deviation
    # Result: each numerical feature has mean=0 and std=1
    numerical_transformer = StandardScaler()

    # ColumnTransformer applies different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer,   numerical_cols),
        ],
        remainder="drop"   # drop any column not in either list
    )

    return preprocessor


def prepare_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Full data preparation pipeline:
      1. Split into X (features) and y (target)
      2. Train/test split — 80% train, 20% test
      3. Fit preprocessor on train data only
      4. Apply preprocessor to both train and test
      5. Apply SMOTE to training data only

    Parameters
    ----------
    df           : cleaned DataFrame from data_loader.py
    test_size    : fraction of data to reserve for testing (default 0.20)
    random_state : for reproducibility

    Returns
    -------
    X_train_res  : resampled (SMOTE) training features
    X_test       : test features (preprocessed, NOT resampled)
    y_train_res  : resampled training labels
    y_test       : test labels
    preprocessor : fitted ColumnTransformer (saved to disk)
    feature_names: list of all feature names after encoding
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Separate features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols   = X.select_dtypes(include=["int64","float64"]).columns.tolist()

    # Train / test split — stratify ensures same class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # important: preserves 23%/77% ratio in both sets
    )
    print(f"[feature_engineering] Train: {len(X_train)}  Test: {len(X_test)}")

    # Build and FIT preprocessor on training data only
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)   # only transform, do not fit

    # Get feature names after one-hot encoding for use in feature importance
    cat_feature_names = preprocessor.named_transformers_["cat"]\
                        .get_feature_names_out(categorical_cols).tolist()
    feature_names = cat_feature_names + numerical_cols

    print(f"[feature_engineering] Features after encoding: {len(feature_names)}")

    # Handle class imbalance with SMOTE
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
        print(f"[feature_engineering] After SMOTE — train: {len(X_train_res)} "
              f"(Left: {y_train_res.sum()}, Stayed: {(y_train_res==0).sum()})")
    else:
        # If SMOTE not available, return original (class_weight handles imbalance)
        X_train_res, y_train_res = X_train_proc, y_train
        print("[feature_engineering] Skipping SMOTE — using class_weight='balanced'")

    # Save fitted preprocessor so predict.py can reuse it
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    print(f"[feature_engineering] Preprocessor saved → {PREPROCESSOR_PATH}")

    return X_train_res, X_test_proc, y_train_res, y_test, preprocessor, feature_names


def encode_user_input(user_dict: dict) -> np.ndarray:
    """
    Convert a dictionary of user-selected values from Gradio into a
    preprocessed numpy array ready for model prediction.

    This function loads the SAVED preprocessor (fitted during training)
    and applies the exact same transformations.

    Parameters
    ----------
    user_dict : dict  e.g.
        {
          "Age": 30,
          "Department": "Sales",
          "OverTime": "Yes",
          "MonthlyIncome": 4500,
          ...
        }

    Returns
    -------
    X : np.ndarray of shape (1, n_features)
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "Preprocessor not found. Run  python src/train.py  first."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Convert dict to single-row DataFrame
    df_input = pd.DataFrame([user_dict])

    # Apply the same transformation as training
    X = preprocessor.transform(df_input)
    return X


# ── UI configuration: what fields to show in the Gradio interface ─────────────
# This dict drives Tab 1 of app.py automatically.
# No need to hard-code inputs in app.py — just add a field here.

UI_FIELDS = {
    # ── Personal Info ──────────────────────────────────────────────────────
    "Age": {
        "type": "slider", "min": 18, "max": 60, "default": 35, "step": 1,
        "label": "Age", "section": "Personal Info"
    },
    "Gender": {
        "type": "radio", "choices": ["Male", "Female"], "default": "Male",
        "label": "Gender", "section": "Personal Info"
    },
    "MaritalStatus": {
        "type": "dropdown",
        "choices": ["Single", "Married", "Divorced"], "default": "Married",
        "label": "Marital Status", "section": "Personal Info"
    },
    "DistanceFromHome": {
        "type": "slider", "min": 1, "max": 29, "default": 5, "step": 1,
        "label": "Distance From Home (km)", "section": "Personal Info"
    },

    # ── Job Info ───────────────────────────────────────────────────────────
    "Department": {
        "type": "dropdown",
        "choices": ["Sales", "Research & Development", "Human Resources"],
        "default": "Research & Development",
        "label": "Department", "section": "Job Info"
    },
    "JobRole": {
        "type": "dropdown",
        "choices": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources"],
        "default": "Research Scientist",
        "label": "Job Role", "section": "Job Info"
    },
    "JobLevel": {
        "type": "slider", "min": 1, "max": 5, "default": 2, "step": 1,
        "label": "Job Level (1=Entry, 5=Senior)", "section": "Job Info"
    },
    "BusinessTravel": {
        "type": "dropdown",
        "choices": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        "default": "Travel_Rarely",
        "label": "Business Travel", "section": "Job Info"
    },
    "OverTime": {
        "type": "radio", "choices": ["Yes", "No"], "default": "No",
        "label": "Works OverTime?", "section": "Job Info"
    },

    # ── Compensation ────────────────────────────────────────────────────────
    "MonthlyIncome": {
        "type": "slider", "min": 1009, "max": 19999, "default": 6500, "step": 100,
        "label": "Monthly Income (PKR equivalent)", "section": "Compensation"
    },
    "PercentSalaryHike": {
        "type": "slider", "min": 11, "max": 25, "default": 14, "step": 1,
        "label": "Last Salary Hike (%)", "section": "Compensation"
    },
    "StockOptionLevel": {
        "type": "slider", "min": 0, "max": 3, "default": 1, "step": 1,
        "label": "Stock Option Level (0=None, 3=High)", "section": "Compensation"
    },

    # ── Satisfaction ────────────────────────────────────────────────────────
    "JobSatisfaction": {
        "type": "slider", "min": 1, "max": 4, "default": 3, "step": 1,
        "label": "Job Satisfaction (1=Low, 4=High)", "section": "Satisfaction"
    },
    "EnvironmentSatisfaction": {
        "type": "slider", "min": 1, "max": 4, "default": 3, "step": 1,
        "label": "Environment Satisfaction (1=Low, 4=High)", "section": "Satisfaction"
    },
    "RelationshipSatisfaction": {
        "type": "slider", "min": 1, "max": 4, "default": 3, "step": 1,
        "label": "Relationship Satisfaction (1=Low, 4=High)", "section": "Satisfaction"
    },
    "WorkLifeBalance": {
        "type": "slider", "min": 1, "max": 4, "default": 3, "step": 1,
        "label": "Work-Life Balance (1=Bad, 4=Best)", "section": "Satisfaction"
    },
    "JobInvolvement": {
        "type": "slider", "min": 1, "max": 4, "default": 3, "step": 1,
        "label": "Job Involvement (1=Low, 4=High)", "section": "Satisfaction"
    },

    # ── Experience ──────────────────────────────────────────────────────────
    "TotalWorkingYears": {
        "type": "slider", "min": 0, "max": 40, "default": 10, "step": 1,
        "label": "Total Working Years", "section": "Experience"
    },
    "YearsAtCompany": {
        "type": "slider", "min": 0, "max": 40, "default": 5, "step": 1,
        "label": "Years at This Company", "section": "Experience"
    },
    "YearsInCurrentRole": {
        "type": "slider", "min": 0, "max": 18, "default": 3, "step": 1,
        "label": "Years in Current Role", "section": "Experience"
    },
    "YearsSinceLastPromotion": {
        "type": "slider", "min": 0, "max": 15, "default": 2, "step": 1,
        "label": "Years Since Last Promotion", "section": "Experience"
    },
    "YearsWithCurrManager": {
        "type": "slider", "min": 0, "max": 17, "default": 4, "step": 1,
        "label": "Years With Current Manager", "section": "Experience"
    },
    "NumCompaniesWorked": {
        "type": "slider", "min": 0, "max": 9, "default": 2, "step": 1,
        "label": "Number of Previous Companies", "section": "Experience"
    },
    "TrainingTimesLastYear": {
        "type": "slider", "min": 0, "max": 6, "default": 3, "step": 1,
        "label": "Training Sessions Last Year", "section": "Experience"
    },

    # ── Education ───────────────────────────────────────────────────────────
    "Education": {
        "type": "slider", "min": 1, "max": 5, "default": 3, "step": 1,
        "label": "Education Level (1=Below College, 5=Doctor)", "section": "Education"
    },
    "EducationField": {
        "type": "dropdown",
        "choices": ["Life Sciences", "Other", "Medical", "Marketing",
                    "Technical Degree", "Human Resources"],
        "default": "Life Sciences",
        "label": "Education Field", "section": "Education"
    },

    # ── Other ────────────────────────────────────────────────────────────
    "PerformanceRating": {
        "type": "radio", "choices": [3, 4], "default": 3,
        "label": "Performance Rating (3=Excellent, 4=Outstanding)", "section": "Other"
    },
    "DailyRate": {
        "type": "slider", "min": 102, "max": 1499, "default": 800, "step": 10,
        "label": "Daily Rate", "section": "Other"
    },
    "HourlyRate": {
        "type": "slider", "min": 30, "max": 100, "default": 65, "step": 1,
        "label": "Hourly Rate", "section": "Other"
    },
    "MonthlyRate": {
        "type": "slider", "min": 2094, "max": 26999, "default": 14000, "step": 100,
        "label": "Monthly Rate", "section": "Other"
    },
}
