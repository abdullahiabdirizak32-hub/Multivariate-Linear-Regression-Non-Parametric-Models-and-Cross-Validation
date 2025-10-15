Practical Lab 2: Multivariate Linear Regression, Non-Parametric Models, and Cross-Validation
# Overview

This project explores predictive modeling of diabetes disease progression using the Scikit-Learn Diabetes dataset.
We compare various regression approaches — both parametric (linear, polynomial) and non-parametric (decision trees, kNN) — to determine which best predicts disease progression one year after baseline.

# Objective

Develop machine learning models to predict diabetes progression scores, based on clinical features such as:
Age, Sex, BMI (Body Mass Index), Blood pressure, Serum measurements

# Dataset Information

The dataset is part of sklearn.datasets:
diabetes = datasets.load_diabetes(as_frame=True)
Number of samples: 442
Number of features: 10 continuous features
Target variable: Disease progression one year after baseline

# Project Structure
## Part 1: Data Preparation and EDA
1. Load and inspect the dataset
2. Framing the problmem
3. Perform Exploratory Data Analysis (EDA):
Descriptive statistics
Histograms and scatter plots
Correlation matrix
4. Check for missing values or duplicates
5. Split dataset into:
75% Training
10% Validation
15% Testing

## Part 2: Univariate Polynomial Regression (BMI vs Target)
6. Fit polynomial regression models (degrees 0 to 5) using the BMI feature.
7. Evaluate using:
R² (Goodness of Fit)
Mean Absolute Error (MAE)
Mean Absolute Percentage Error (MAPE)
8. Identify the best model using validation performance.
9. Test the chosen model on Test data.
10. Plot the model fit over train/validation/test sets.
11. Display the model equation and prediction example.
12. Predict the target variable
13. Display the trainable parameters
14. Key learning outcomes:
Understand how model complexity affects bias and variance.
Visualize polynomial fits and overfitting behavior.

## Part 3: Multivariate Models
selected ones to build and compare:
Two Polynomial Regression models (e.g., degrees 2 and 3)
1. Two Decision Tree models (varying max_depth)
2. Two k-Nearest Neighbors (kNN) models (varying n_neighbors)
3. One Linear Regression baseline
4. Each model is evaluated on R², MAE, and MAPE using the train-validation-test pipeline.