"""
app.py
------
Gradio web interface for the Employee Attrition Prediction System.

Tab 1 — Predict Attrition
    HR fills in employee details using dropdowns, sliders, and radio buttons.
    System predicts: Will Leave / Will Stay + confidence + risk level + top 3 reasons.

Tab 2 — Model Performance
    Shows accuracy table, ROC curve chart, confusion matrices, feature importance.

Run:
    python app.py
    → open http://localhost:7860
"""

import os
import sys
import json
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict            import (predict_attrition, format_result,
                                     get_available_models, get_best_model)
from src.feature_engineering import UI_FIELDS

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ── Load metadata ─────────────────────────────────────────────────────────────
if not os.path.exists(META_PATH):
    raise RuntimeError(
        "Models not found!\n"
        "Run this command first:  python src/train.py\n"
        "Then run:                python app.py"
    )

with open(META_PATH) as f:
    META = json.load(f)

AVAILABLE_MODELS = get_available_models()
BEST_MODEL       = get_best_model()


# ── Confidence gauge chart ────────────────────────────────────────────────────

def _make_gauge(probability: float) -> plt.Figure:
    """
    Draw a semicircular gauge chart showing attrition probability.
    Green = low risk, Yellow = medium, Red = high.
    """
    fig, ax = plt.subplots(figsize=(4, 2.5),
                            subplot_kw={"projection": "polar"})

    # Draw background arcs
    theta_total = np.pi  # semicircle
    segments = [
        (0,           np.pi*0.40, "#4CAF50", "Low"),
        (np.pi*0.40,  np.pi*0.70, "#FF9800", "Medium"),
        (np.pi*0.70,  np.pi,      "#F44336", "High"),
    ]
    for start, end, color, _ in segments:
        theta = np.linspace(start, end, 50)
        ax.fill_between(theta, 0.7, 1.0, color=color, alpha=0.3)
        ax.plot(theta, [1.0]*50, color=color, linewidth=4)

    # Draw needle
    needle_angle = np.pi * (1 - probability)
    ax.annotate("", xy=(needle_angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=2.5, mutation_scale=15))

    # Labels
    ax.text(np.pi/2, 0.35, f"{probability*100:.1f}%",
            ha="center", va="center", fontsize=16, fontweight="bold",
            transform=ax.transData)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1.1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_title("Attrition Risk", fontsize=11, pad=6)

    plt.tight_layout()
    return fig


# ── Prediction handler ────────────────────────────────────────────────────────

def run_prediction(model_name, *slider_values):
    """
    Called when the user clicks 'Predict'.
    Collects all Gradio input values, runs prediction, returns formatted outputs.
    """
    # Zip field names with values from Gradio
    field_names = list(UI_FIELDS.keys())
    user_dict   = {}

    for fname, val in zip(field_names, slider_values):
        cfg = UI_FIELDS[fname]
        # Convert slider integers to int type for rule-based matching
        if cfg["type"] == "slider":
            user_dict[fname] = int(val) if "." not in str(val) else float(val)
        else:
            user_dict[fname] = val

    try:
        result               = predict_attrition(user_dict, model_name=model_name)
        headline, reasons_md, details = format_result(result)
        gauge                = _make_gauge(result["probability"])

        return headline, reasons_md, details, gauge

    except FileNotFoundError as e:
        msg = f"❌ {e}"
        return msg, "", "", None
    except Exception as e:
        msg = f"❌ Unexpected error: {e}"
        return msg, "", "", None


# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    theme = gr.themes.Soft(
        primary_hue   = "blue",
        secondary_hue = "slate",
        neutral_hue   = "slate",
        font          = gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="Employee Attrition Predictor", theme=theme) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown(
            f"""
            # 👔 Employee Attrition Prediction System
            **IBM HR Analytics · Classical ML · {len(AVAILABLE_MODELS)} Models Trained**

            Fill in the employee's details below and click **Predict** to find out
            whether the employee is likely to leave the company, along with the
            confidence level, risk rating, and top reasons.
            """
        )

        # ── Model selector ─────────────────────────────────────────────────
        with gr.Row():
            model_selector = gr.Radio(
                choices   = AVAILABLE_MODELS,
                value     = BEST_MODEL,
                label     = "🤖 Select Model",
                info      = f"Recommended: {BEST_MODEL} (highest F1 score)"
            )

        gr.Markdown("---")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 1 — PREDICT
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Predict Attrition"):

            # Group fields by section
            sections = {}
            for fname, cfg in UI_FIELDS.items():
                sec = cfg.get("section", "Other")
                sections.setdefault(sec, []).append((fname, cfg))

            input_components = []   # collect in order for run_prediction()

            # ── Input fields: full width, grouped by section ──────────────
            for section_name, fields in sections.items():
                gr.Markdown(f"#### {section_name}")
                with gr.Row():
                    for fname, cfg in fields:
                        if cfg["type"] == "slider":
                            comp = gr.Slider(
                                minimum   = cfg["min"],
                                maximum   = cfg["max"],
                                step      = cfg["step"],
                                value     = cfg["default"],
                                label     = cfg["label"],
                                interactive = True,
                            )
                        elif cfg["type"] == "dropdown":
                            comp = gr.Dropdown(
                                choices   = cfg["choices"],
                                value     = cfg["default"],
                                label     = cfg["label"],
                                interactive = True,
                            )
                        elif cfg["type"] == "radio":
                            comp = gr.Radio(
                                choices   = cfg["choices"],
                                value     = cfg["default"],
                                label     = cfg["label"],
                                interactive = True,
                            )
                        input_components.append(comp)

            predict_btn = gr.Button(
                "🔮 Predict Attrition", variant="primary", size="lg"
            )

            gr.Markdown("---")

            # ── Result panel: BELOW all input fields ──────────────────────
            gr.Markdown("#### 📊 Prediction Result")
            out_headline = gr.Markdown("_Fill in details and click Predict_")

            with gr.Row():
                with gr.Column(scale=1):
                    out_gauge   = gr.Plot(label="Risk Gauge")
                with gr.Column(scale=2):
                    out_reasons = gr.Markdown()
                    out_details = gr.Markdown()

            # ── Quick presets ──────────────────────────────────────────────
            gr.Markdown("---")
            gr.Markdown("#### ⚡ Quick Test Presets")
            with gr.Row():
                preset_high = gr.Button("🔴 High Risk Employee")
                preset_low  = gr.Button("🟢 Low Risk Employee")
                preset_mid  = gr.Button("🟡 Medium Risk Employee")

            # High risk preset: overtime, low satisfaction, young, single
            HIGH_RISK = {
                "Age":30,"Gender":"Male","MaritalStatus":"Single",
                "DistanceFromHome":22,"Department":"Sales",
                "JobRole":"Sales Representative","JobLevel":1,
                "BusinessTravel":"Travel_Frequently","OverTime":"Yes",
                "MonthlyIncome":2800,"PercentSalaryHike":11,"StockOptionLevel":0,
                "JobSatisfaction":1,"EnvironmentSatisfaction":1,
                "RelationshipSatisfaction":2,"WorkLifeBalance":1,
                "JobInvolvement":2,"TotalWorkingYears":3,"YearsAtCompany":1,
                "YearsInCurrentRole":0,"YearsSinceLastPromotion":0,
                "YearsWithCurrManager":0,"NumCompaniesWorked":5,
                "TrainingTimesLastYear":0,"Education":3,
                "EducationField":"Marketing","PerformanceRating":3,
                "DailyRate":300,"HourlyRate":35,"MonthlyRate":3000,
            }
            # Low risk preset: senior, married, high pay, satisfied
            LOW_RISK = {
                "Age":45,"Gender":"Female","MaritalStatus":"Married",
                "DistanceFromHome":4,"Department":"Research & Development",
                "JobRole":"Research Director","JobLevel":4,
                "BusinessTravel":"Non-Travel","OverTime":"No",
                "MonthlyIncome":18000,"PercentSalaryHike":22,"StockOptionLevel":3,
                "JobSatisfaction":4,"EnvironmentSatisfaction":4,
                "RelationshipSatisfaction":4,"WorkLifeBalance":4,
                "JobInvolvement":4,"TotalWorkingYears":20,"YearsAtCompany":15,
                "YearsInCurrentRole":8,"YearsSinceLastPromotion":1,
                "YearsWithCurrManager":10,"NumCompaniesWorked":1,
                "TrainingTimesLastYear":5,"Education":5,
                "EducationField":"Life Sciences","PerformanceRating":4,
                "DailyRate":1200,"HourlyRate":90,"MonthlyRate":22000,
            }
            # Medium risk preset
            MED_RISK = {
                "Age":32,"Gender":"Male","MaritalStatus":"Single",
                "DistanceFromHome":12,"Department":"Sales",
                "JobRole":"Sales Executive","JobLevel":2,
                "BusinessTravel":"Travel_Rarely","OverTime":"Yes",
                "MonthlyIncome":5000,"PercentSalaryHike":13,"StockOptionLevel":1,
                "JobSatisfaction":2,"EnvironmentSatisfaction":3,
                "RelationshipSatisfaction":3,"WorkLifeBalance":2,
                "JobInvolvement":3,"TotalWorkingYears":7,"YearsAtCompany":4,
                "YearsInCurrentRole":2,"YearsSinceLastPromotion":3,
                "YearsWithCurrManager":2,"NumCompaniesWorked":3,
                "TrainingTimesLastYear":2,"Education":3,
                "EducationField":"Marketing","PerformanceRating":3,
                "DailyRate":600,"HourlyRate":55,"MonthlyRate":10000,
            }

            def _load_preset(preset: dict):
                return [preset.get(k, UI_FIELDS[k]["default"]) for k in UI_FIELDS]

            preset_high.click(fn=lambda: _load_preset(HIGH_RISK), outputs=input_components)
            preset_low.click( fn=lambda: _load_preset(LOW_RISK),  outputs=input_components)
            preset_mid.click( fn=lambda: _load_preset(MED_RISK),  outputs=input_components)

            # Wire predict button
            predict_btn.click(
                fn      = run_prediction,
                inputs  = [model_selector] + input_components,
                outputs = [out_headline, out_reasons, out_details, out_gauge],
            )

        # ═══════════════════════════════════════════════════════════════════
        # TAB 2 — MODEL PERFORMANCE
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Model Performance"):
            gr.Markdown("### All Models — Performance Comparison")

            # Metrics table
            rows = [
                [name,
                 f"{v['accuracy']}%",
                 f"{v['precision']}%",
                 f"{v['recall']}%",
                 f"{v['f1']}%",
                 f"{v['roc_auc']}"]
                for name, v in META["results"].items()
            ]
            gr.Dataframe(
                value   = rows,
                headers = ["Model","Accuracy","Precision","Recall","F1","ROC-AUC"],
                label   = "Evaluation Metrics (20% test set)",
                interactive = False,
            )

            # Charts
            with gr.Row():
                comp_path = os.path.join(MODEL_DIR, "model_comparison.png")
                if os.path.exists(comp_path):
                    gr.Image(value=comp_path, label="Model Comparison")

                roc_path = os.path.join(MODEL_DIR, "roc_curves.png")
                if os.path.exists(roc_path):
                    gr.Image(value=roc_path, label="ROC Curves")

            # Best model confusion matrix and feature importance
            gr.Markdown(f"### {BEST_MODEL} — Detailed Results")
            with gr.Row():
                cm_path = os.path.join(MODEL_DIR, f"{BEST_MODEL}_confusion.png")
                if os.path.exists(cm_path):
                    gr.Image(value=cm_path, label=f"Confusion Matrix — {BEST_MODEL}")

                fi_path = os.path.join(MODEL_DIR, f"{BEST_MODEL}_importance.png")
                if os.path.exists(fi_path):
                    gr.Image(value=fi_path, label=f"Feature Importance — {BEST_MODEL}")

            gr.Markdown(
                """
                ### Metric Definitions

                **Accuracy** — percentage of all predictions that are correct.

                **Precision** — of all employees predicted to leave, what % actually left?
                High precision = fewer false alarms.

                **Recall** — of all employees who actually left, what % did we catch?
                High recall = we miss fewer real resignations. This is the MOST IMPORTANT
                metric for HR — missing someone who will leave is costly.

                **F1 Score** — harmonic mean of precision and recall. Best single metric
                for imbalanced datasets like this one.

                **ROC-AUC** — measures overall model discrimination. 1.0 = perfect,
                0.5 = random guessing. A value above 0.85 is excellent.
                """
            )

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            "<center><sub>👔 Employee Attrition Predictor · "
            "IBM HR Dataset · scikit-learn + Gradio · "
            "BS Computer Science Project</sub></center>"
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        show_error  = True,
    )