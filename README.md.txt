# Customer Churn Prediction

## 📌 Project Overview

This project predicts customer churn for a telecommunications company using machine learning. The goal is to identify customers likely to cancel their service, enabling proactive retention strategies. The work covers the full data science pipeline: data cleaning, exploratory analysis, feature engineering, model training, evaluation, and business recommendations.

## 📊 Dataset

- **Source**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Samples**: 7,043 customers
- **Features**: 21 (demographics, account info, services subscribed, charges)
- **Target**: `Churn` – Yes (1) / No (0)

## 🛠️ Tools & Libraries

| Category       | Libraries                                                                 |
|----------------|---------------------------------------------------------------------------|
| Data handling  | `pandas`, `numpy`                                                         |
| Visualization  | `matplotlib`, `seaborn`                                                   |
| Preprocessing  | `sklearn.preprocessing` (LabelEncoder, StandardScaler)                   |
| Modeling       | `sklearn.linear_model` (LogisticRegression), `sklearn.ensemble` (RandomForestClassifier) |
| Evaluation     | `sklearn.metrics` (accuracy, confusion matrix, classification report)    |

## 🧠 Workflow

1. **Data Loading & Inspection** – Load CSV, check data types, missing values.
2. **Data Cleaning** – Drop `customerID`, convert `TotalCharges` to numeric, drop null rows, map Churn to 0/1.
3. **Feature Encoding** – Apply `LabelEncoder` to all categorical features.
4. **Exploratory Data Analysis**:
   - Class distribution (imbalanced)
   - Box plots: Monthly Charges vs Churn, Tenure vs Churn
   - Correlation heatmap
5. **Preprocessing for Modeling** – Separate features (X) and target (y), scale features with `StandardScaler`.
6. **Train/Test Split** – 80/20 split (`random_state=42`).
7. **Model Training**:
   - Logistic Regression (`max_iter=1000`)
   - Random Forest (`n_estimators=200`)
8. **Evaluation** – Accuracy, confusion matrix, precision/recall/f1-score.
9. **Feature Importance** – Top 10 features from Random Forest.
10. **Business Insights & Recommendations**.

## 📈 Results

| Model                | Accuracy | Precision (Churn) | Recall (Churn) | F1-score (Churn) |
|----------------------|----------|------------------|----------------|------------------|
| Logistic Regression  | 78.5%    | 0.62             | 0.49           | 0.55             |
| Random Forest        | 78.8%    | 0.64             | 0.48           | 0.54             |

Both models perform similarly. Random Forest has slightly higher accuracy, but Logistic Regression has marginally better recall. The **imbalanced nature** of the dataset (27% churn) makes recall more critical – further improvements could use class weighting or SMOTE.

## 🔍 Key Insights (from Random Forest)

Top 5 most important features:
1. **TotalCharges** (18.7%)
2. **MonthlyCharges** (17.8%)
3. **tenure** (15.5%)
4. **Contract** (8.0%)
5. **PaymentMethod** (5.1%)

## 💡 Business Recommendations

- Offer **discounts for long-term contracts** (e.g., 1-year or 2-year plans).
- Target **high-monthly-charge customers** with loyalty programs or service reviews.
- Monitor **new customers with short tenure** – they are more likely to churn.
- Improve **online security and tech support** – these features show moderate importance.
