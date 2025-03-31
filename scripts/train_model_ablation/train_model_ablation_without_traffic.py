"""
train_model_ablation_without_traffic.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 24/03/2025
Description: Train XGBoost model without 'traffic_congestion_level' to evaluate feature impact.

This script follows the ablation study proposed in the thesis to assess the effect
of removing key variables identified in the feature importance analysis.
It is based on train_model.py, with the key difference being the removal of one
feature (traffic_congestion_level) to analyze its impact on model performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from scripts.preprocessing.load_data import load_data
import numpy as np

# Load processed data from MariaDB (same as in train_model.py)
df = load_data()

if df is None or df.empty:
    raise ValueError("Error: No data loaded. Check database connection and preprocessing.")

# Define target and remove specific feature for ablation (DIFFERENCE from train_model.py)
TARGET = "fuel_consumption_rate"
FEATURE_TO_REMOVE = "traffic_congestion_level"

if TARGET not in df.columns or FEATURE_TO_REMOVE not in df.columns:
    raise ValueError("Error: Required variables not found in dataset.")

# Drop the target and the selected feature to simulate ablation
features = df.drop(columns=[TARGET, FEATURE_TO_REMOVE])

# Drop non-numeric or problematic columns if necessary
for col in features.columns:
    if pd.api.types.is_datetime64_any_dtype(features[col]) or features[col].dtype == 'object':
        features.drop(columns=[col], inplace=True)

labels = df[TARGET]

# Train-test split (same as train_model.py)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train XGBoost model (same as train_model.py, but using fewer features)
print("Training XGBoost model WITHOUT feature:", FEATURE_TO_REMOVE)
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate performance (same metrics as in train_model.py)
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Print results for comparison
print("\nResults - XGBoost without", FEATURE_TO_REMOVE)
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
