"""
src/train.py
------------
Responsibility: Train all 6 ML models and save them to the models/ folder.

Models trained:
  1. Logistic Regression  — linear baseline, very interpretable
  2. Decision Tree        — rule-based, easy to explain in viva
  3. Random Forest        — ensemble of 300 trees, best overall
  4. SVM                  — strong boundary-finder, good with scaled data
  5. KNN                  — distance-based, simple and intuitive
  6. Naive Bayes          — probabilistic baseline, very fast

Evaluation metrics used:
  - Accuracy   : overall correct predictions
  - Precision  : of all predicted "Leave", how many actually left?
  - Recall     : of all employees who left, how many did we catch?
  - F1-Score   : harmonic mean of precision and recall (best for imbalanced data)
  - ROC-AUC    : area under ROC curve — 1.0 is perfect, 0.5 is random

Why F1 and ROC-AUC instead of just accuracy?
  If 77% of employees stay, a model that always says "Stay" gets 77% accuracy.
  That is a useless model. F1 and AUC measure whether the model actually
  identifies the employees who will LEAVE — which is the real goal.

Run this file ONCE before launching app.py:
    python src/train.py
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

warnings.filterwarnings("ignore")

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader        import load_raw_data
from src.feature_engineering import prepare_data

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ── Step 1: Define all models ─────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dictionary of all 6 classifiers with their configurations.

    class_weight='balanced' tells models to give MORE importance to the
    minority class (Attrition=1). This is a second line of defense against
    class imbalance, on top of SMOTE.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,              # limit depth to prevent overfitting
            class_weight="balanced",
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,         # 300 trees → stable, accurate predictions
            max_depth=None,           # trees grow fully
            class_weight="balanced",
            n_jobs=-1,                # use all CPU cores
            random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",             # RBF handles non-linear boundaries
            C=1.0,
            gamma="scale",
            probability=True,         # needed for predict_proba
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,            # 7 neighbours — odd number avoids ties
            metric="euclidean",
            weights="distance"        # closer neighbours have more voting power
        ),
        "NaiveBayes": GaussianNB()
        # NaiveBayes does not support class_weight —
        # SMOTE handles imbalance for it
    }


# ── Step 2: Train and evaluate each model ─────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test, feature_names: list):
    """
    Train all 6 models, evaluate them, save each one, and produce comparison charts.

    Parameters
    ----------
    X_train, X_test  : preprocessed feature arrays
    y_train, y_test  : 0/1 label arrays
    feature_names    : list of feature names for importance plots

    Returns
    -------
    results : dict  {model_name: {accuracy, precision, recall, f1, roc_auc}}
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    models  = get_models()
    results = {}

    print("\n" + "="*65)
    print("  EMPLOYEE ATTRITION — MODEL TRAINING")
    print("="*65)
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-"*65)

    for name, model in models.items():

        # ── Train ──────────────────────────────────────────────────────────
        model.fit(X_train, y_train)

        # ── Predict ────────────────────────────────────────────────────────
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]   # probability of "Leave"

        # ── Metrics ────────────────────────────────────────────────────────
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)

        print(f"  {name:<22} {acc*100:>5.1f}% {prec*100:>5.1f}% "
              f"{rec*100:>5.1f}% {f1*100:>5.1f}% {auc:>6.3f}")

        results[name] = {
            "accuracy":  round(acc  * 100, 2),
            "precision": round(prec * 100, 2),
            "recall":    round(rec  * 100, 2),
            "f1":        round(f1   * 100, 2),
            "roc_auc":   round(auc,  3),
        }

        # ── Save model ─────────────────────────────────────────────────────
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)

        # ── Confusion matrix ───────────────────────────────────────────────
        _save_confusion_matrix(y_test, y_pred, name)

        # ── ROC curve (collect for combined plot) ──────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name]["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        # ── Feature importance (tree-based models only) ────────────────────
        if hasattr(model, "feature_importances_"):
            _save_feature_importance(model, feature_names, name)

    print("-"*65)

    # ── Find best model by F1 score ────────────────────────────────────────
    best_name = max(
        [n for n in results],
        key=lambda n: results[n]["f1"]
    )
    print(f"\n  Best model (F1): {best_name} "
          f"— F1={results[best_name]['f1']}%  AUC={results[best_name]['roc_auc']}")

    return results, best_name


# ── Step 3: Save metadata and charts ──────────────────────────────────────────

def save_metadata(results: dict, best_name: str, feature_names: list):
    """Save training results summary to models/metadata.json."""
    # Remove ROC curve data from metadata (too large, saved separately)
    clean_results = {
        name: {k: v for k, v in vals.items() if k != "roc"}
        for name, vals in results.items()
    }
    metadata = {
        "best_model":    best_name,
        "model_names":   list(results.keys()),
        "feature_names": feature_names,
        "results":       clean_results,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved → models/metadata.json")


def _save_confusion_matrix(y_true, y_pred, model_name: str):
    """Save a confusion matrix heatmap for one model."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Stayed (0)", "Left (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_confusion.png"), dpi=130)
    plt.close()


def _save_feature_importance(model, feature_names: list, model_name: str):
    """Save a horizontal bar chart of top-15 feature importances."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]
    top_names   = [feature_names[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_names[::-1], top_vals[::-1], color="#E65100")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top-15 Feature Importances — {model_name}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_importance.png"), dpi=130)
    plt.close()


def save_comparison_chart(results: dict):
    """Save a grouped bar chart comparing all models across all metrics."""
    models  = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "AUC×100"]
    colors  = ["#1565C0","#2E7D32","#F57F17","#880E4F","#4A148C"]

    # Scale AUC to percentage for chart readability
    data = []
    for m in metrics:
        row = []
        for name in models:
            val = results[name][m]
            row.append(val * 100 if m == "roc_auc" else val)
        data.append(row)

    x     = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (vals, label, color) in enumerate(zip(data, labels, colors)):
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6.5
            )

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Employee Attrition Prediction", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("  Comparison chart saved → models/model_comparison.png")


def save_roc_curves(results: dict):
    """Save a combined ROC curve plot for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1565C0","#2E7D32","#F57F17","#880E4F","#4A148C","#00838F"]

    for (name, vals), color in zip(results.items(), colors):
        fpr = vals["roc"]["fpr"]
        tpr = vals["roc"]["tpr"]
        auc = vals["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, linewidth=1.8)

    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print("  ROC curves saved → models/roc_curves.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  STEP 1 — Loading data")
    print("="*65)
    df = load_raw_data()

    print("\n" + "="*65)
    print("  STEP 2 — Feature engineering")
    print("="*65)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    print("\n" + "="*65)
    print("  STEP 3 — Training models")
    print("="*65)
    results, best_name = train_all(X_train, X_test, y_train, y_test, feature_names)

    print("\n" + "="*65)
    print("  STEP 4 — Saving results")
    print("="*65)
    save_metadata(results, best_name, feature_names)
    save_comparison_chart(results)
    save_roc_curves(results)

    print("\n" + "="*65)
    print("  TRAINING COMPLETE")
    print(f"  Best model : {best_name}")
    print(f"  Models dir : models/")
    print("  Next step  : python app.py")
    print("="*65 + "\n")
