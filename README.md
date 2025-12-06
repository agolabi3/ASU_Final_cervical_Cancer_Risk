🧬 **Cervical Cancer Risk Prediction**

CIS 508 – Machine Learning in Business • Final Project

Author: Aryanna Golabi

XGBoost App: https://asufinalcervicalcancerrisk-l3xd56p3d3fkwnokbxdtkp.streamlit.app/

Decision Tree App: https://asufinalcervicalcancerrisk-ymgdfdb9e2jmkrzupmef5c.streamlit.app/

Course: CIS 508 — Machine Learning in Business

📌 **Project Overview**

Cervical cancer is one of the most preventable cancers when detected early. The goal of this project is to build a machine-learning tool that predicts the risk of a positive cervical cancer biopsy using demographic, behavioral, and clinical features.

This project demonstrates the full ML lifecycle:

Business problem framing

Data cleaning & preprocessing

Model development with hyperparameter tuning

MLflow experiment tracking in Databricks

Deployment as interactive Streamlit apps

Two models were ultimately deployed:

XGBoost model – nonlinear, higher-capacity classifier

Decision Tree model – simple, interpretable model that achieved the best evaluation metrics

🏥 **1. Business Problem & Analytical Framing
Business Problem**

Clinicians often face limited time and resources for screening. A data-driven risk prediction tool can help:

Identify patients who may require more urgent follow-up

Reduce missed diagnoses

Improve screening efficiency

Why It Matters

False negatives in cancer screening are high-stakes. Therefore:

Recall (Sensitivity) was selected as the primary metric.

The project frames the problem as a binary classification task:

Target variable: Biopsy

1 = positive (evidence of cervical cancer)

0 = negative

ML Approach

Supervised learning

Models tested: LR, SVM, Decision Tree, Random Forest, KNN, Naive Bayes, Neural Network, XGBoost, Ensemble

Full hyperparameter tuning across all models

Best model selected by Recall, with Precision/Accuracy considered for clinical viability

🧹 **2. Data Preparation**
Dataset

Source: Kaggle – Cervical Cancer Risk Factors Data Set

~850 rows

Includes demographics, history, behavioral factors, and clinical diagnoses

Cleaning Steps

Replace ? with NaN

Convert numeric columns

Drop rows missing biopsy outcomes

Median imputation for all features

Scaling using StandardScaler

Train/test split: 80% / 20% with stratification

Features Used in Modeling

Age, number of partners, first intercourse age

Number of pregnancies

Smoking status, smoking years, packs/year

Hormonal contraceptive use + years

IUD use + years

STDs (presence, number)

Clinical flags: Dx:Cancer, Dx:CIN, Dx:HPV

🤖 **3. Model Development & Hyperparameter Tuning**

All required models for CIS 508 were implemented and tuned:

Models Built

Logistic Regression

Support Vector Machine

Decision Tree

Random Forest

K-Nearest Neighbors

Naive Bayes

Neural Network (MLP)

XGBoost

Voting Ensemble (LR + RF + SVM)

Hyperparameter Tuning

Each model had 2–3 hyperparameters, each with multiple tested values:

Example:

Decision Tree: max_depth, min_samples_split, criterion

XGBoost: n_estimators, learning_rate, max_depth

MLP: hidden_layer_sizes, alpha, activation

SVM: C, kernel, gamma

This generated 100+ MLflow experiment runs.

MLflow Tracking (Databricks)

Tracked per run:

Model family

Hyperparameters

Accuracy

Precision

Recall

F1 score

Model artifact

🏆 **4. Best-Performing Model**
❌ Why Naive Bayes was not selected

Although Naive Bayes achieved Recall = 1.0, it had:

Precision ≈ 0.06

Accuracy ≈ 0.10

It was predicting everyone as positive.
Clinically useless → rejected.

⭐ **True Best Model: Decision Tree (max_depth=3)**

Based on the actual MLflow results:

Recall: 0.8182

Precision: 0.75

Accuracy: ~0.97

F1 Score: ~0.78

Why the Decision Tree was selected

Best balance of Recall and Precision

Highest overall Accuracy of all high-recall models

Small, interpretable structure

Clinically reasonable predictions

Based on simple, stable decision rules

This model is the official “best” model for the project based on your experiments.

🚀 **5. Deployment** (Two Streamlit Apps)
**1. XGBoost App**

Demonstrates deployment of an advanced nonlinear ML model

Retrains XGBoost inside the app

Uses full feature set

Provides probability estimates

Good example of modern applied ML deployment

**2. Decision Tree App (Best Model)**

Implements the true best model

Transparent, interpretable decision structure

Ideal for clinical-style reasoning

Shows probability and classification

More stable to user variation

Better aligned to metric priority (Recall)

Satisfies professor’s emphasis on interpretability

📁 **Project Structure**
ASU_Cervical_Cancer_Risk/
│
├── streamlit_app.py                 # XGBoost app
├── streamlit_dt_app.py              # Decision Tree app – BEST MODEL
├── asu_cervical_cancer_risk_final.py (Colab modeling script)
├── risk_factors_cervical_cancer.csv
├── requirements.txt
└── README.md

📊 **6. Key Insights**
Clinical insights

Prior diagnoses (Dx:CIN, Dx:HPV, Dx:Cancer) strongly correlate with positive biopsy

STD history + smoking amplify risk but do not dominate the tree

Shallow trees provide clinician-friendly decision logic

Business value

Supports early screening workflows

Prioritizes patients needing urgent follow-up

Reduces missed diagnoses (Recall-first design)

Deployable as a lightweight web tool

⚠️ **Medical Disclaimer**

This tool is for educational purposes only as part of CIS 508.
It should not be used for real clinical decision-making.

🎉 **Conclusion**

This project successfully demonstrates:

Full ML lifecycle

Hyperparameter tuning across many models

Databricks MLflow experiment tracking

Two deployed web apps

Interpretability-focused model selection

Strong business and clinical alignment
