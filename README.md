🧬 **Cervical Cancer Risk Prediction**

CIS 508 – Machine Learning in Business • Final Project

Author: Aryanna Golabi

App: https://asufinalcervicalcancerrisk-l3xd56p3d3fkwnokbxdtkp.streamlit.app/

Course: CIS 508 — Machine Learning in Business

📌 **Project Overview**

Cervical cancer is largely preventable through early detection and regular screening. However, many individuals lack access to timely medical follow-up, making risk‐prediction tools valuable in prioritizing screenings.

This project builds a machine learning model that predicts the likelihood of a positive cervical cancer biopsy using behavioral, demographic, and clinical risk factors.

This README documents:

The business problem

Analytic framing and ML approach

Data preparation

Model development with extensive hyperparameter tuning

MLflow experiment tracking

Deployment using Streamlit

How to run and use the application

🏥 **1. Business Problem & Analytical Framing
Business Problem**

Healthcare providers often face high patient volumes and limited resources. A data-driven screening tool can help prioritize individuals at highest risk for cervical cancer to ensure timely follow-up and reduce missed diagnoses.

Why It Matters

Cervical cancer outcomes improve dramatically with early detection.

False negatives are dangerous — therefore Recall (Sensitivity) is the priority metric.

A risk prediction tool can support clinical decision-making and public health initiatives.

ML Task Framing

This is a supervised binary classification problem:

Target variable: Biopsy

1 = positive cervical cancer biopsy

0 = negative biopsy

ML Goal: Predict likelihood of a positive biopsy

Metric priority: Recall, to avoid missed true cancer cases

🧹 **2. Data Preparation
Dataset**

Source: Cervical Cancer Risk Classification Dataset (Kaggle)
File: risk_factors_cervical_cancer.csv

Key Steps

Handle missing values (?) → replaced using median imputation

Convert all numeric features

Drop rows without biopsy outcomes

Feature scaling using StandardScaler

Train-test split: 80% train / 20% test with stratification

Features Used

A clinically meaningful set, including:

Age

Number of sexual partners

Age at first intercourse

Number of pregnancies

Smoking history (status, years, packs/year)

Hormonal contraceptives (use + years)

IUD use

STD history + count

Prior diagnoses (Cancer, CIN, HPV)

🤖 **3. Model Development & Hyperparameter Tuning**

All models required by the professor were implemented and tracked using MLflow in Databricks, with at least two hyperparameters tuned for each model:

Models Tested
Model Family	Hyperparameters Tuned
Logistic Regression	C, class_weight, penalty
Support Vector Machine	C, kernel, gamma
Decision Tree	max_depth, min_samples_split, criterion
Random Forest	n_estimators, max_depth, max_features
K-Nearest Neighbors	n_neighbors, weights, p
Naive Bayes	var_smoothing
Gradient Boosting / XGBoost	n_estimators, learning_rate, max_depth
Neural Network (MLP)	hidden_layer_sizes, alpha, activation
Voting Ensemble	voting, weights, flatten_transform
Tracking in MLflow

Every run logs:

Model family

Hyperparameters

Accuracy, Precision, Recall, F1 score

Serialized model artifact

Model Selection

The best performing model (by Recall) was:

👉 XGBoost Classifier, tuned with:

n_estimators=200

learning_rate=0.1

max_depth=5

This model achieved the highest sensitivity in validation and was therefore selected for deployment.

🚀 **4. Deployment (Streamlit App)**
App Features

Built using Streamlit

Users input patient history and behavioral risk factors

App preprocesses data exactly like the trained model

Model provides:

High/Low risk classification

Probability of positive biopsy

Explanatory context & medical disclaimer

Run the App Locally
pip install -r requirements.txt
streamlit run streamlit_app.py

Requirements
streamlit==1.39.0
numpy
pandas
scikit-learn
xgboost

📁 Project File Structure
ASU_Final_cervical_Cancer_Risk/
│
├── streamlit_app.py               # Deployed Streamlit web app
├── asu_cervical_cancer_risk_final.py  # Colab modeling script
├── risk_factors_cervical_cancer.csv   # Dataset
├── requirements.txt
└── README.md                      # This document

🔬 **5. Key Insights & Business Value
Findings**

Behavioral and sexual health history strongly influence risk prediction.

Smoking duration and STD history are particularly predictive.

XGBoost outperformed all other models, capturing nonlinear interactions.

Business Impact

A screening support tool like this could help medical staff identify high-risk patients earlier.

Prioritizing patients with elevated predicted risk could reduce missed cancer diagnoses.

The model emphasizes Recall, ensuring fewer false negatives in a clinical context.

⚠️ **Medical Disclaimer**

This tool is not a medical device, diagnosis tool, or substitute for clinical judgment.
It is intended only for educational purposes as part of an academic project.

🎉 **Acknowledgments**

This project was completed for CIS 508 – Machine Learning in Business,
Arizona State University.
