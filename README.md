# Loan Default Prediction Using Machine Learning

Predicting loan default risk using multiple machine learning models and comparing their predictive performance, efficiency, and interpretability.

---

## Project Overview

This project develops and compares multiple machine learning models to predict loan default risk using the Kaggle Loan Default Prediction dataset.

The project demonstrates an end-to-end machine learning workflow, including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model development
- Hyperparameter tuning
- Cross-validation
- Performance evaluation
- Model interpretation using SHAP

The objective is to identify high-risk borrowers and support data-driven lending decisions for financial institutions.

---

## Business Problem

Loan default prediction is a binary classification problem commonly encountered in the financial industry.

Accurately identifying borrowers with a high probability of default enables financial institutions to:

- Reduce credit risk
- Improve lending decisions
- Minimize financial losses
- Enhance risk management strategies

---

## Dataset

**Dataset**

credit_risk_dataset.csv

**Source**

Kaggle Playground Series – Season 4 Episode 10

**Target Variable**

- loan_status

The dataset contains borrower demographic information, financial characteristics, loan details, and historical credit records for predicting loan default risk.

---

## Technologies

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- LightGBM
- SHAP

---

# Project Workflow

## 1. Data Preprocessing

The following preprocessing steps were performed before model training:

- Verified that the dataset contained no missing values
- Removed applicants older than 100 years
- Removed records where employment started before age 14
- Converted categorical variables into numerical representations:
  - person_home_ownership
  - loan_intent
  - loan_grade
  - cb_person_default_on_file
- Removed **loan_int_rate** because of its strong correlation with **loan_grade** to reduce multicollinearity.

---

## 2. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand feature distributions and relationships.

Main analyses include:

- Histograms for feature distributions
- Boxplots for outlier detection
- Correlation analysis
- Correlation heatmap

The analysis showed that **loan_int_rate** was highly correlated with **loan_grade**, so the feature was removed before model training.

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

---

## 3. Model Development

The following machine learning models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network
- Gradient Boosting Decision Tree (GBDT)
- XGBoost
- LightGBM

Model optimization included:

- Hyperparameter tuning
- Grid Search
- 5-Fold Cross Validation

---

# Model Performance

| Model | Training Accuracy | Test Accuracy |
|------|------------------:|--------------:|
| Logistic Regression | 89.12% | 89.45% |
| Decision Tree | 94.35% | 94.14% |
| Random Forest | 92.67% | 93.08% |
| Neural Network | 92.98% | 86.06% |
| GBDT | 95.82% | **95.07%** |
| XGBoost | **96.31%** | 95.00% |
| LightGBM | 95.07% | 94.71% |

---

# Classification Metrics

| Model | Precision | Recall | F1 Score |
|------|----------:|--------:|---------:|
| Logistic Regression | 74.34% | 38.53% | 50.76% |
| Decision Tree | 87.10% | 68.56% | 76.73% |
| Random Forest | 91.58% | 56.15% | 69.62% |
| Neural Network | 84.32% | 64.17% | 72.88% |
| GBDT | **90.72%** | **72.51%** | **80.60%** |
| XGBoost | 90.45% | 72.19% | 80.30% |
| LightGBM | 88.72% | 71.66% | 79.29% |

---

# Runtime Comparison

| Model | Runtime (seconds) |
|------|------------------:|
| Logistic Regression | 0.88 |
| Decision Tree | 0.12 |
| Random Forest | 2.05 |
| Neural Network | 51.51 |
| GBDT | 5.38 |
| XGBoost | 0.20 |
| LightGBM | **0.08** |

---

## ROC Curve Comparison

The ROC curves compare the predictive performance of different machine learning models.

GBDT achieved the highest predictive accuracy, while XGBoost produced nearly identical performance with significantly faster training. LightGBM offered the best balance between prediction accuracy and computational efficiency.

![ROC Curve](images/roc_curve.png)

---

# Model Interpretation

To improve model transparency, SHAP (SHapley Additive exPlanations) was applied to tree-based models.

The SHAP analysis showed that:

- **person_income** was the most influential feature affecting model predictions.
- **loan_grade** was the second most important predictor.
- **person_home_ownership** and **loan_amnt** also contributed significantly.
- **cb_person_cred_hist_length** and **cb_person_default_on_file** had relatively small impacts on prediction outcomes.

SHAP helped explain how each feature influenced individual predictions and improved the interpretability of the machine learning models.

---

# Key Findings

- Ensemble learning models consistently outperformed traditional machine learning algorithms.
- GBDT achieved the best predictive performance with **95.07%** testing accuracy and an **80.60%** F1-score.
- XGBoost delivered nearly identical predictive performance while requiring significantly less training time.
- LightGBM achieved the fastest runtime while maintaining competitive predictive accuracy.
- Correlation analysis reduced multicollinearity by removing **loan_int_rate** before model training.
- SHAP analysis identified **person_income** and **loan_grade** as the most influential features.

---

# Repository Structure

```
Loan-Default-Prediction
│
├── README.md
├── Final_Project.ipynb
├── Final_Project.html
├── requirements.txt
│
├── images
│   ├── correlation_heatmap.png
│   └── roc_curve.png
│
└── models
```

---

# Future Improvements

- Apply feature scaling and transform skewed variables where appropriate.
- Improve preprocessing by handling outliers and feature distributions more systematically.
- Improve model interpretability and fairness in lending decisions.
- Deploy the best-performing model as a web application using Streamlit or Flask.
- Build an automated prediction pipeline and monitoring system for model drift.

---

# Author

**Chenyin Luo**

M.S. in Data Analytics  
Tufts University

GitHub: https://github.com/chenyin9808
