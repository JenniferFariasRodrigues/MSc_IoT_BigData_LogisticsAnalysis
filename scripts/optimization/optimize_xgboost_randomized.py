"""
optimize_xgboost_randomized.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 01/04/2025
Description: Hyperparameter tuning of XGBoost using RandomizedSearchCV.

Objective: Improve fuel consumption prediction accuracy by refining model parameters.
This script performs randomized hyperparameter search to improve the predictive
performance of the XGBoost model for fuel consumption forecasting. It builds upon
the original training pipeline used in train_model.py, adding two searches over a
predefined parameter grid and selecting the best model configurations based on
cross-validated MAE and MAPE. The selected models are then evaluated on the test set
using MAE, RMSE, and MAPE metrics.
"""

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from scripts.preprocessing.load_data import load_data

# === Load dataset and prepare features ===
df = load_data()
TARGET = "fuel_consumption_rate"
if TARGET not in df.columns:
    raise ValueError("Target column not found.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Drop non-numeric columns if necessary
X = X.select_dtypes(include=[np.number])

# === Step 1: Split the dataset into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 2: Define the hyperparameter search space ===
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# === Step 3: Function to run RandomizedSearchCV and evaluate results ===
def run_search(scorer_name):
    print(f"\n===== Running RandomizedSearchCV using scoring: {scorer_name} =====")
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring=scorer_name,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    print("Best hyperparameters found:", search.best_params_)
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
    return {
        'scoring': scorer_name,
        'params': search.best_params_,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

# === Step 4: Run both searches and compare ===
results_mae = run_search('neg_mean_absolute_error')
results_mape = run_search('neg_mean_absolute_percentage_error')

# === Summary ===
print("\n===== Summary of Hyperparameter Tuning Results =====")
for res in [results_mae, results_mape]:
    print(f"\nScoring: {res['scoring']}")
    print(f"Best Params: {res['params']}")
    print(f"MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f} | MAPE: {res['mape']:.2f}%")



# """
# optimize_xgboost_randomized.py
# --------------------------------------------------
# Author: Jennifer Farias Rodrigues
# Date: 01/04/2025
# Description: Hyperparameter tuning of XGBoost using RandomizedSearchCV.
#
# Objective: Improve fuel consumption prediction accuracy by refining model parameters.
# This script performs randomized hyperparameter search to improve the predictive
# performance of the XGBoost model for fuel consumption forecasting. It builds upon
# the original training pipeline used in train_model.py, adding a search over a
# predefined parameter grid and selecting the best model configuration based on
# cross-validated MAE. The selected model is then evaluated on the test set using
# MAE, RMSE, and MAPE metrics.
# """
#
#
# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# from sklearn.model_selection import train_test_split
# import numpy as np
# from scripts.preprocessing.load_data import load_data
#
# # === Load dataset and prepare features ===
# df = load_data()
# TARGET = "fuel_consumption_rate"
# if TARGET not in df.columns:
#     raise ValueError("Target column not found.")
#
# X = df.drop(columns=[TARGET])
# y = df[TARGET]
#
# # Drop non-numeric columns if necessary
# X = X.select_dtypes(include=[np.number])
#
# # === Step 1: Split the dataset into training and testing sets ===
# # Using 80% for training and 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # === Step 2: Define the hyperparameter search space for XGBoost ===
# # These parameters control the model complexity and regularization
# param_grid = {
#     'n_estimators': [50, 100, 200],            # Number of boosting rounds
#     'max_depth': [3, 5, 7, 10],                # Maximum depth of a tree
#     'learning_rate': [0.01, 0.05, 0.1, 0.3],   # Step size shrinkage
#     'subsample': [0.6, 0.8, 1.0],              # Fraction of observations used per tree
#     'colsample_bytree': [0.6, 0.8, 1.0],       # Fraction of features used per tree
#     'reg_alpha': [0, 0.1, 1],                  # L1 regularization term
#     'reg_lambda': [1, 1.5, 2]                  # L2 regularization term
# }
#
# # === Step 3: Initialize the base XGBoost model ===
# xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
#
# # === Step 4: Apply RandomizedSearchCV to find the best parameters ===
# # We use 5-fold cross-validation and negative MAE as the scoring metric
# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_grid,
#     n_iter=50,               # Number of parameter combinations to try
#     scoring='neg_mean_absolute_error',
#     cv=5,                    # 5-fold cross-validation
#     verbose=1,               # Display progress
#     random_state=42,
#     n_jobs=-1                # Use all available cores
# )
#
# # === Step 5: Fit the model to the training data ===
# random_search.fit(X_train, y_train)
#
# # === Step 6: Predict and evaluate the optimized model ===
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)
#
# # Compute evaluation metrics
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mape = mean_absolute_percentage_error(y_test, y_pred) * 100
#
# # Display results
# print("Best hyperparameters found:", random_search.best_params_)
# print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
