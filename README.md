# 📉 Customer Churn Prediction
> Predicting telecom customer churn with interpretable machine learning and translating model outputs into actionable retention recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EE4C2C?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-7C3AED?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=flat-square)

---

## 🎯 Problem Statement

A telecom company wants to identify customers at risk of cancelling their service (**churn**) before it happens, in order to apply targeted retention strategies.

**Business question:** *Which customers are most likely to leave, and what are the main business drivers behind churn?*

---

## 📦 Dataset

**[Telco Customer Churn — IBM Watson / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

| | |
|---|---|
| Rows | 7,043 customers |
| Features | 21 original features |
| Target | `Churn` (Yes / No) |
| Positive class | ~26.5% |
| Challenge | Class imbalance + business-oriented recall optimization |

---

## 🔍 Project Structure

```text
churn-prediction/
│
├── churn_prediction.ipynb
├── README.md
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── img/
    ├── categorical_churn_rates.png
    ├── churn_distribution.png
    ├── confusion_matrix.png
    ├── numerical_distributions.png
    ├── roc_pr_curves.png
    ├── shap_force_plot.png
    ├── shap_summary.png
    └── threshold_optimization.png
```

---

## 🧠 Approach

### 1. Exploratory Data Analysis
- Checked class balance and target distribution
- Compared numerical feature distributions by churn
- Measured churn rates across key categorical variables

### 2. Feature Engineering
- `AvgMonthlySpend = TotalCharges / (tenure + 1)`
- `IsNewCustomer = tenure <= 6`
- `HasMultipleServices = count of selected service add-ons`

### 3. Preprocessing
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `ColumnTransformer` to keep preprocessing reproducible

### 4. Modeling
Three models were compared with stratified cross-validation:

| Model | CV ROC-AUC |
|---|---:|
| **Logistic Regression** | **0.8486** |
| Random Forest | 0.8261 |
| XGBoost | 0.8229 |

### 5. Threshold Optimization
Because missing a churner is more costly than contacting a non-churner, the final decision threshold was optimized with **F2-score**, which weights recall more heavily than precision.

### 6. Interpretability
- Global SHAP summary plot
- Individual SHAP explanation for a churn case
- The current notebook uses `shap.LinearExplainer` with the final Logistic Regression model

---

## 📊 Key Results

Final predictive model: **Logistic Regression**

| Metric | Value |
|---|---:|
| Test ROC-AUC | **0.8472** |
| Average Precision | **0.6603** |
| Optimal threshold (F2) | **0.58** |
| F2-score | **0.7595** |
| Recall (Churn) | **0.9358** |
| Precision (Churn) | **0.4332** |

Interpretation:
- Logistic Regression delivered the strongest overall ranking performance on both ROC-AUC and Average Precision.
- After F2 threshold optimization, the model captures most churners, which is useful when recall matters more than precision.
- The precision tradeoff suggests retention actions should be relatively low-cost or tiered by risk level.

---

## 💡 Business Insights

Based on EDA and SHAP analysis, the most actionable churn drivers are:

| Driver | Finding | Recommended Action |
|---|---|---|
| **Contract type** | Month-to-month customers show by far the highest churn rate compared with one- and two-year contracts | Offer migration incentives toward longer-term plans |
| **Tenure** | The first 6 months are the most fragile stage of the customer lifecycle | Launch an onboarding and early-retention program |
| **Monthly charges** | Higher monthly charges are associated with greater churn risk | Test discounts, loyalty credits, or plan-rightsizing for high-bill customers |
| **Tech Support / Online Security** | Customers without these services churn substantially more | Offer targeted trials or bundled service benefits |
| **Payment method** | Electronic check customers show the highest churn rate | Encourage autopay or card-based payment migration |
| **Internet service** | Fiber optic customers show higher churn than DSL customers | Review pricing, service quality, and premium-segment retention offers |

---

## 🚀 How to Run

```bash
git clone https://github.com/<yasmineperez>/churn-prediction.git
cd churn-prediction

pip install -r requirements.txt

jupyter notebook churn_prediction.ipynb
```

---

## 👩‍💻 Author

**Yasmine Pérez**  
[LinkedIn](https://www.linkedin.com/in/yasmine-perez-97b233292/) · [GitHub](https://github.com/yasmineperez)
