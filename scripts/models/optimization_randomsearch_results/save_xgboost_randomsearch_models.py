"""
save_xgboost_randomsearch_models.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 01/04/2025
Description: Save XGBoost models optimized via RandomizedSearchCV.

This script loads and trains two XGBoost models using RandomizedSearchCV:
1. One using 'neg_mean_absolute_error' (MAE) as the scoring metric.
2. One using 'neg_mean_absolute_percentage_error' (MAPE) as the scoring metric.

Both optimized models are evaluated and saved in the
'models/optimization_randomsearch_results/' directory in .pkl format for reproducibility and deployment.
"""

import os
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scripts.preprocessing.load_data import load_data

# === Load data ===
df = load_data()
TARGET = "fuel_consumption_rate"
if TARGET not in df.columns:
    raise ValueError("Target column not found.")

X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define search space ===
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# === Function to train and evaluate ===
def run_random_search(scoring_method):
    print(f"\n===== Running RandomizedSearchCV using scoring: {scoring_method} =====")
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=50,
        scoring=scoring_method,
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

    return best_model, search.best_params_, mae, rmse, mape

# === Run for both scoring strategies ===
best_model_mae, best_params_mae, mae_mae, rmse_mae, mape_mae = run_random_search('neg_mean_absolute_error')
best_model_mape, best_params_mape, mae_mape, rmse_mape, mape_mape = run_random_search('neg_mean_absolute_percentage_error')

# === Save models ===
save_dir = os.path.join("models", "optimization_randomsearch_results")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_model_mae, os.path.join(save_dir, "xgboost_randomsearch_mae.pkl"))
joblib.dump(best_model_mape, os.path.join(save_dir, "xgboost_randomsearch_mape.pkl"))

# === Summary Report ===
print("\n===== Summary of Hyperparameter Tuning Results =====")
print(f"\nScoring: neg_mean_absolute_error\nBest Params: {best_params_mae}\nMAE: {mae_mae:.4f} | RMSE: {rmse_mae:.4f} | MAPE: {mape_mae:.2f}%")
print(f"\nScoring: neg_mean_absolute_percentage_error\nBest Params: {best_params_mape}\nMAE: {mae_mape:.4f} | RMSE: {rmse_mape:.4f} | MAPE: {mape_mape:.2f}%")
