"""
save_xgboost_gridsearch_model.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 01/04/2025
Description: Save XGBoost model optimized via GridSearchCV (MAPE scoring).

This script runs GridSearchCV using a refined hyperparameter grid to optimize
XGBoost for fuel consumption prediction, using MAPE as the scoring function.
The best model is saved to the directory 'models/optimization_gridsearch_results/'.
"""

import os
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scripts.preprocessing.load_data import load_data

# === Load data ===
df = load_data()
TARGET = "fuel_consumption_rate"
if TARGET not in df.columns:
    raise ValueError("Target column not found.")

X = df.drop(columns=[TARGET])
y = df[TARGET]
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define parameter grid ===
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1.0, 1.5]
}

# === Run GridSearchCV ===
print("===== Running GridSearchCV using scoring: neg_mean_absolute_percentage_error =====")
model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_percentage_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# === Output Results ===
print("Best Parameters:", grid_search.best_params_)
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

# === Save the model ===
output_dir = os.path.join("models", "optimization_gridsearch_results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "xgboost_gridsearch_mape.pkl")
joblib.dump(best_model, output_path)
print(f"\nModel saved to: {output_path}")
