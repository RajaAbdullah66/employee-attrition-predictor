# 👔 Employee Attrition Prediction System

> IBM HR Analytics Dataset · Classical ML · 6 Models · Gradio Interface

Predict whether an employee will leave the company using classical machine
learning algorithms. Built as a BS Computer Science final-year project.

---

## Project Structure

```
employee_attrition/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   ← IBM HR dataset (download from Kaggle)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← loads CSV, cleans data, encodes target
│   ├── feature_engineering.py  ← encodes categories, scales numbers, SMOTE
│   ├── train.py                ← trains 6 models, saves charts + .joblib files
│   └── predict.py              ← inference + risk level + top reasons
├── models/                     ← auto-created by train.py
│   ├── RandomForest.joblib
│   ├── LogisticRegression.joblib
│   ├── ... (all 6 models)
│   ├── preprocessor.joblib
│   ├── metadata.json
│   ├── roc_curves.png
│   └── model_comparison.png
├── app.py                      ← Gradio web interface
├── requirements.txt
└── README.md
```

---

## How to Get the Dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. Click **Download**
3. Extract the zip file
4. Copy `WA_Fn-UseC_-HR-Employee-Attrition.csv` into the `data/` folder

---

## Setup & Run

```bash
# Step 1: Install all dependencies
pip install -r requirements.txt

# Step 2: Train all 6 models (run once)
python src/train.py

# Step 3: Launch the web app
python app.py
# → Open browser at http://localhost:7860
```

---

## Dataset

| Property | Value |
|---|---|
| Source | IBM HR Analytics (Kaggle) |
| Rows | 1,470 employees |
| Features | 35 columns |
| Target | Attrition (Yes=Left / No=Stayed) |
| Class split | ~84% Stay / ~16% Leave |
| Imbalance fix | SMOTE + class_weight='balanced' |

---

## Features Used

**Personal:** Age, Gender, MaritalStatus, DistanceFromHome

**Job:** Department, JobRole, JobLevel, BusinessTravel, OverTime

**Compensation:** MonthlyIncome, PercentSalaryHike, StockOptionLevel

**Satisfaction:** JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement, RelationshipSatisfaction

**Experience:** TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, NumCompaniesWorked

---

## Models & Pipeline

```
Raw Input → OneHotEncoder (categorical) + StandardScaler (numerical)
         → SMOTE (balance classes)
         → Classifier
         → Attrition Prediction
```

| Model | Why Used |
|---|---|
| Logistic Regression | Linear baseline, very interpretable |
| Decision Tree | Rule-based, easy to visualise |
| Random Forest | Best ensemble, handles non-linearity |
| SVM | Strong boundary finder, good with scaled data |
| KNN | Distance-based, intuitive |
| Naive Bayes | Probabilistic baseline |

---

## Prediction Output

- **Will Leave / Will Stay**
- **Confidence %** (probability from model)
- **Risk Level**: 🟢 Low (<40%) · 🟡 Medium (40–70%) · 🔴 High (>70%)
- **Top 3 Reasons** (rule-based explainability)
- **HR Action Recommendations**

