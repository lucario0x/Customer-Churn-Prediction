# Customer Churn Prediction

Predicting customer churn on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset — with a focus on doing the evaluation properly for an imbalanced classification problem, not just reporting accuracy.

## Why this project

Churn is a naturally imbalanced problem (~73% No-churn vs ~27% Churn here). A model that predicts "No churn" for every customer already scores ~73% accuracy while being useless. This project is built around that constraint: proper preprocessing, imbalance-aware models, cross-validated metrics beyond accuracy, hyperparameter tuning, and explainability — then turning the model output into something a retention team could actually act on.

## Dataset

- **Source:** Telco Customer Churn (7,043 customers, 21 columns)
- **Target:** `Churn` (Yes/No)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method, charges), and subscribed services (internet, phone, streaming, tech support, etc.)

## Approach

1. **EDA** — churn distribution, correlation heatmap, tenure/monthly-charges distributions by churn, churn rate by contract type and payment method.
2. **Preprocessing** — categorical columns are one-hot encoded and numeric columns are scaled inside a single `sklearn` `Pipeline`, fit only on the training fold to avoid data leakage.
3. **Model comparison** — Logistic Regression, Decision Tree, Random Forest, KNN, and XGBoost, compared using 5-fold stratified cross-validation on **Accuracy, Precision, Recall, F1, and ROC-AUC** (not accuracy alone). Imbalance is handled with `class_weight='balanced'` / `scale_pos_weight`.
4. **Hyperparameter tuning** — `RandomizedSearchCV` on the strongest candidate (XGBoost), optimizing for ROC-AUC.
5. **Explainability** — SHAP values to show *which* features drive risk up or down for individual customers, not just global feature importance.
6. **From model to action** — test-set customers ranked by `churn probability × monthly charges`, to prioritize retention outreach toward high-risk, high-value customers.

## Results

Cross-validated comparison (mean over 5 folds):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.752 | 0.522 | 0.803 | 0.632 | **0.846** |
| Random Forest | 0.790 | 0.642 | 0.478 | 0.547 | 0.824 |
| XGBoost | 0.769 | 0.556 | 0.651 | 0.599 | 0.822 |
| K-Nearest Neighbors | 0.756 | 0.543 | 0.520 | 0.531 | 0.778 |
| Decision Tree | 0.729 | 0.490 | 0.486 | 0.488 | 0.652 |

**Tuned XGBoost — held-out test set:**
- ROC-AUC: **0.839**
- F1 (churn class): **0.614**
- Recall (churn class): 0.79 — catches most customers who actually churn, at the cost of some false positives (a reasonable trade-off for retention outreach, where missing a churner is costlier than an extra check-in call).

## Tech stack

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`

## Project structure

```
├── Customer_Churn_Prediction_Model.ipynb   # full analysis notebook
├── Telco_Customer_Churn.csv                # dataset
├── churn_model_pipeline.pkl                # saved, tuned model pipeline
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib
jupyter notebook Customer_Churn_Prediction_Model.ipynb
```
