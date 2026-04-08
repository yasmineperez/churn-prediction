# 📉 Customer Churn Prediction
> Predicting telecom customer churn with interpretable ML — and translating model outputs into actionable business recommendations.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EE4C2C?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-7C3AED?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=flat-square)

---

## 🎯 Problem Statement

A telecom company wants to identify customers at risk of cancelling their service (**churn**) before it happens, in order to apply targeted retention strategies.

**Business question:** *Which customers are most likely to leave next month — and why?*

---

## 📦 Dataset

**[Telco Customer Churn — IBM Watson / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

| | |
|---|---|
| Rows | 7,043 customers |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` (Yes / No) — ~27% positive class |
| Challenge | Class imbalance |

---

## 🔍 Project Structure

```
churn-prediction/
│
├── churn_prediction.ipynb      # Full analysis notebook
├── requirements.txt
├── README.md
│
└── img/                        # Saved plots
    ├── churn_distribution.png
    ├── numerical_distributions.png
    ├── categorical_churn_rates.png
    ├── roc_pr_curves.png
    ├── threshold_optimization.png
    ├── confusion_matrix.png
    ├── shap_summary.png
    └── shap_force_plot.png
```

---

## 🧠 Approach

### 1. Exploratory Data Analysis
- Class imbalance analysis (~27% churn rate)
- Distribution of numerical features (tenure, monthly charges, total charges) segmented by churn
- Churn rate per categorical feature (contract type, payment method, services)

### 2. Feature Engineering
- `AvgMonthlySpend` = TotalCharges / (tenure + 1)
- `IsNewCustomer` = tenure ≤ 6 months
- `HasMultipleServices` = count of contracted add-ons

### 3. Preprocessing
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `ColumnTransformer` pipeline for clean train/test isolation

### 4. Modeling & Comparison

| Model | CV AUC-ROC |
|---|---|
| Logistic Regression | ~0.845 |
| Random Forest | ~0.862 |
| **XGBoost** | **~0.873** ✅ |

- `class_weight='balanced'` / `scale_pos_weight` to handle imbalance
- `StratifiedKFold` cross-validation

### 5. Threshold Optimization
Default threshold (0.5) minimizes false negatives in this context — but in churn, **missing a churner is more costly** than a false alarm. Threshold was optimized using **F2-score** (which weights Recall twice as much as Precision).

### 6. Model Interpretability (SHAP)
- Global feature importance (summary plot)
- Individual prediction explanations (force plot)

---

## 📊 Key Results

| Metric | Value |
|---|---|
| AUC-ROC | **0.87** |
| Recall (Churn class) | **~0.79** |
| Precision (Churn class) | **~0.63** |
| Optimal threshold | **0.35** |

---

## 💡 Business Insights

| Driver | Finding | Recommended Action |
|---|---|---|
| **Contract type** | Month-to-month → 3x higher churn | Offer incentives to switch to annual plans |
| **Tenure** | First 6 months are critical | Early onboarding program + proactive check-ins |
| **Monthly charges** | High charge + low tenure = high risk | Bundle discounts for high-spend new customers |
| **Tech Support** | No support services → higher churn | Free trial of support add-ons |
| **Payment method** | Electronic check → highest churn rate | Incentivize auto-pay adoption |

### 💰 Estimated Business Impact
Assuming ~7,000 customers, 27% churn, $65 avg monthly revenue:
- If the model captures **70% of actual churners** and **40%** are retained with intervention
- → Estimated savings: **~$34K/month**

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yasmineperez/churn-prediction.git
cd churn-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place it in the root folder:
# WA_Fn-UseC_-Telco-Customer-Churn.csv

# Create img folder
mkdir img

# Run the notebook
jupyter notebook churn_prediction.ipynb
```

---

## 📚 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
jupyter
```

---

## 👩‍💻 Author

**Yasmine Pérez** — Computer Engineer, USM Chile  
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/yasmineperez)

---

*Part of my [Data Science Portfolio](https://github.com/yasmineperez)*
