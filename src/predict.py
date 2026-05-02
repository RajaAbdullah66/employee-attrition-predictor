"""
src/predict.py
--------------
Responsibility: Load a saved model and make predictions on new employee data.

Two functions:
  predict_attrition(user_dict, model_name)
    → Takes user input from Gradio as a dictionary
    → Returns prediction, confidence, risk level, and top reasons

  get_top_reasons(user_dict, model_name)
    → Identifies the top 3 features contributing to attrition risk
    → Used to explain WHY the model thinks the employee might leave

Risk levels:
  Low    : confidence < 40%   → Green  → probably safe
  Medium : 40% ≤ conf < 70%  → Yellow → monitor this employee
  High   : confidence ≥ 70%  → Red    → intervention needed

Why do we show reasons?
  Telling HR "this employee might leave" is not enough.
  Telling HR "this employee works overtime, has low satisfaction, and earns
  below average" gives them something ACTIONABLE to fix.
  This is called explainability — making ML decisions understandable.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import encode_user_input

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Cache ─────────────────────────────────────────────────────────────────────
_model_cache:    dict = {}
_metadata_cache: dict = {}


def _load_metadata() -> dict:
    """Load metadata.json once and cache it."""
    global _metadata_cache
    if not _metadata_cache:
        path = os.path.join(MODEL_DIR, "metadata.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "metadata.json not found. Run  python src/train.py  first."
            )
        with open(path) as f:
            _metadata_cache = json.load(f)
    return _metadata_cache


def _load_model(model_name: str):
    """Load a saved model from disk. Cache in memory after first load."""
    if model_name not in _model_cache:
        path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {path}.\n"
                "Run  python src/train.py  first."
            )
        _model_cache[model_name] = joblib.load(path)
    return _model_cache[model_name]


def get_available_models() -> list:
    """Return list of all trained model names."""
    return _load_metadata().get("model_names", [
        "LogisticRegression","DecisionTree","RandomForest","SVM","KNN","NaiveBayes"
    ])


def get_best_model() -> str:
    """Return the name of the best-performing model."""
    return _load_metadata().get("best_model", "RandomForest")


# ── Risk labelling ─────────────────────────────────────────────────────────────

def _get_risk_level(probability: float) -> tuple:
    """
    Convert a raw probability (0–1) into a human-readable risk label.

    Returns (label, emoji, color_hint)
    """
    if probability < 0.40:
        return "Low Risk",    "🟢", "green"
    elif probability < 0.70:
        return "Medium Risk", "🟡", "orange"
    else:
        return "High Risk",   "🔴", "red"


# ── Reason generation ─────────────────────────────────────────────────────────

# These are rule-based reasons grounded in real HR research.
# Each tuple: (feature_name, condition_fn, reason_text)
ATTRITION_RULES = [
    ("OverTime",                 lambda v: v == "Yes",   "Works overtime — high burnout risk"),
    ("JobSatisfaction",          lambda v: v <= 2,        "Low job satisfaction (rated ≤ 2/4)"),
    ("WorkLifeBalance",          lambda v: v == 1,        "Poor work-life balance (rated 1/4)"),
    ("MonthlyIncome",            lambda v: v < 3500,      "Below-average monthly income"),
    ("YearsAtCompany",           lambda v: v < 3,         "Less than 3 years at company — early turnover risk"),
    ("Age",                      lambda v: v < 28,        "Young employee — higher mobility tendency"),
    ("MaritalStatus",            lambda v: v == "Single", "Single — more likely to relocate or switch"),
    ("EnvironmentSatisfaction",  lambda v: v <= 2,        "Low environment satisfaction (rated ≤ 2/4)"),
    ("StockOptionLevel",         lambda v: v == 0,        "No stock options — lower financial retention"),
    ("BusinessTravel",           lambda v: v == "Travel_Frequently", "Frequent business travel — exhausting"),
    ("JobInvolvement",           lambda v: v <= 2,        "Low job involvement — disengaged employee"),
    ("NumCompaniesWorked",       lambda v: v >= 6,        "Has worked at many companies — job hopper"),
    ("YearsSinceLastPromotion",  lambda v: v >= 5,        "No promotion in 5+ years — stagnation"),
    ("DistanceFromHome",         lambda v: v >= 20,       "Lives far from office — commute fatigue"),
    ("RelationshipSatisfaction", lambda v: v == 1,        "Very poor relationship satisfaction"),
    ("TrainingTimesLastYear",    lambda v: v == 0,        "No training last year — feels underinvested"),
]


def get_top_reasons(user_dict: dict, top_n: int = 3) -> list:
    """
    Return the top N reasons why this employee might leave,
    based on rule-based analysis of their feature values.

    Parameters
    ----------
    user_dict : dict of user-selected feature values
    top_n     : how many reasons to return (default 3)

    Returns
    -------
    list of reason strings, e.g.:
      ["Works overtime — high burnout risk",
       "Low job satisfaction (rated ≤ 2/4)",
       "Below-average monthly income"]
    """
    triggered = []
    for feature, condition, reason in ATTRITION_RULES:
        value = user_dict.get(feature)
        if value is not None:
            try:
                if condition(value):
                    triggered.append(reason)
            except (TypeError, ValueError):
                pass

    # If fewer than top_n rules triggered, add a generic note
    if len(triggered) < top_n:
        triggered.append("Profile is generally stable — low attrition indicators")

    return triggered[:top_n]


# ── Main prediction function ───────────────────────────────────────────────────

def predict_attrition(user_dict: dict, model_name: str = None) -> dict:
    """
    Predict whether an employee will leave the company.

    Parameters
    ----------
    user_dict   : dict of feature values from the Gradio UI
    model_name  : which model to use (default: best model from metadata)

    Returns
    -------
    dict with keys:
      prediction    : "Will Leave" or "Will Stay"
      probability   : float 0–1 (probability of leaving)
      confidence_pct: float 0–100
      risk_level    : "Low Risk", "Medium Risk", or "High Risk"
      risk_emoji    : "🟢", "🟡", or "🔴"
      top_reasons   : list of 3 reason strings
      model_used    : name of the model
    """
    if model_name is None:
        model_name = get_best_model()

    # Step 1: Encode user input using the saved preprocessor
    X = encode_user_input(user_dict)

    # Step 2: Load model and predict
    model  = _load_model(model_name)
    pred   = model.predict(X)[0]          # 0 = Stay, 1 = Leave
    proba  = model.predict_proba(X)[0]    # [prob_stay, prob_leave]
    prob_leave = float(proba[1])

    # Step 3: Risk level
    risk_label, risk_emoji, _ = _get_risk_level(prob_leave)

    # Step 4: Top reasons
    reasons = get_top_reasons(user_dict, top_n=3)

    return {
        "prediction":     "🚨 Will Leave" if pred == 1 else "✅ Will Stay",
        "probability":    prob_leave,
        "confidence_pct": round(prob_leave * 100, 1),
        "risk_level":     risk_label,
        "risk_emoji":     risk_emoji,
        "top_reasons":    reasons,
        "model_used":     model_name,
        "raw_pred":       int(pred),
    }


def format_result(result: dict) -> tuple:
    """
    Format prediction result into strings for Gradio display.

    Returns
    -------
    (headline_md, reasons_md, details_md)
    """
    pred   = result["prediction"]
    conf   = result["confidence_pct"]
    risk   = result["risk_level"]
    emoji  = result["risk_emoji"]
    model  = result["model_used"]
    reasons = result["top_reasons"]

    # Headline
    headline = (
        f"## {pred}\n\n"
        f"**Attrition Probability:** {conf:.1f}%\n\n"
        f"**Risk Level:** {emoji} {risk}\n\n"
        f"**Model Used:** `{model}`"
    )

    # Top reasons
    reasons_md = "### 🔍 Top Reasons\n\n"
    icons = ["🔴", "🟠", "🟡"]
    for i, reason in enumerate(reasons):
        reasons_md += f"{icons[i]} {reason}\n\n"

    # HR advice
    if result["raw_pred"] == 1:
        advice = (
            "### 💼 Recommended HR Actions\n\n"
            "- Schedule a one-on-one meeting with the employee\n"
            "- Review compensation against market benchmarks\n"
            "- Discuss career growth and promotion timeline\n"
            "- Consider workload reduction if overtime is high\n"
            "- Offer flexible working hours or remote options\n"
        )
    else:
        advice = (
            "### 💼 HR Notes\n\n"
            "- Employee appears satisfied — maintain current engagement\n"
            "- Continue regular performance reviews\n"
            "- Ensure training and development opportunities are available\n"
        )

    details = advice

    return headline, reasons_md, details
