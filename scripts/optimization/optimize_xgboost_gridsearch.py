"""
optimize_xgboost_gridsearch.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 01/04/2025
Description: Hyperparameter tuning of XGBoost using GridSearchCV.

Objective: Improve fuel consumption prediction accuracy by refining model parameters.
This script performs an exhaustive grid search to optimize the predictive performance
of the XGBoost model for fuel consumption forecasting, focusing on reducing MAPE.
The best model is selected based on cross-validated MAPE and evaluated on the test set.
"""

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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

# === Step 2: Define a refined hyperparameter grid for GridSearchCV ===
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1.0, 1.5]
}

# === Step 3: Initialize and configure GridSearchCV ===
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_percentage_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# === Step 4: Train and evaluate ===
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# === Step 5: Print results ===
print("\n===== GridSearchCV Results (MAPE scoring) =====")
print("Best Parameters:", grid_search.best_params_)
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
