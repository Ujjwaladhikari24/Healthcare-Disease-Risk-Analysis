# Healthcare Disease Risk Analysis

## Project Overview

Healthcare Disease Risk Analysis is a machine learning based system designed to analyze patient health and lifestyle information and estimate diabetes risk.

The project combines data analysis, preprocessing, feature engineering, classification, statistical analysis, visualization, model evaluation, risk scoring, and a Flask-based web application.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Flask
- Joblib
- HTML
- CSS

## Main Features

- Patient health data analysis
- Data preprocessing
- Feature engineering
- Diabetes classification
- Risk score generation
- Missing optional data handling
- Model comparison
- Model evaluation
- Confusion matrix
- Feature importance analysis
- Diabetes class distribution visualization
- Healthcare analytics dashboard
- Flask web application

## Machine Learning Models

Two classification algorithms are evaluated:

1. Logistic Regression
2. Random Forest Classifier

The final model is selected using balanced accuracy to account for class imbalance.

## Prediction Classes

The system predicts three categories:

- Healthy
- Prediabetes
- Diabetes

## Risk Score

The system generates a model-based risk score from 0 to 100 based on the predicted probability of prediabetes or diabetes.

Risk categories:

- 0–29: Low Risk
- 30–59: Moderate Risk
- 60–100: High Risk

## Dataset

The project uses the CDC BRFSS diabetes health indicators dataset.

The target variable is:

`Diabetes_012`

## Project Structure

Healthcare-Disease-Risk-Analysis/

├── Data/

├── Models/

├── Notebooks/

├── Static/

├── Templates/

├── train_model.py

├── app.py

├── requirements.txt

└── README.md

## Model Training

The training process includes:

1. Dataset loading
2. Data cleaning
3. Duplicate removal
4. Missing-value handling
5. Train-test splitting
6. Feature preprocessing
7. Logistic Regression training
8. Random Forest training
9. Model evaluation
10. Best-model selection
11. Confusion matrix generation
12. Feature importance analysis
13. Model serialization

## Evaluation Metrics

The project evaluates:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Web Application

The Flask application provides:

- Patient assessment form
- Optional-field handling
- Disease prediction
- Risk score
- Risk category
- Prediction probabilities
- Analytics dashboard

## Optional Information

Optional patient information can be left blank.

Missing optional values are automatically replaced using values derived from the training dataset.

Important health features are required for a prediction.

## Educational Disclaimer

This project is developed for academic and educational purposes.

The predictions and risk scores are not medical diagnoses and should not be used as a substitute for professional medical advice.