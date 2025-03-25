"""
train_model.py
--------------
Author: Jennifer Farias Rodrigues
Date: 10/03/2025
Description: Train predictive models for IoT logistics optimization.

Structure:
- Load processed data from MariaDB
- Split dataset into train (80%) and test (20%)
- Train multiple models (ARIMA, Random Forest, XGBoost)
- Save trained models for evaluation
"""

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from scripts.load_data import load_data  # Import processed data

# Load processed data
df = load_data()

# Ensure data is loaded correctly
if df is None or df.empty:
    raise ValueError("Error: No data loaded. Check database connection and data processing.")

# Define target variable and features
target = "fuel_consumption_rate"

if target not in df.columns:
    raise ValueError(f"Error: Target variable '{target}' not found in dataset columns: {df.columns}")

features = df.drop(columns=[target])

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

# Ensure the models directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train, y_train)

# Save models
rf_model_path = os.path.join(model_dir, "random_forest.pkl")
xgb_model_path = os.path.join(model_dir, "xgboost.pkl")

with open(rf_model_path, "wb") as f:
    pickle.dump(rf_model, f)

with open(xgb_model_path, "wb") as f:
    pickle.dump(xgb_model, f)

print(f"Models trained and saved successfully in {model_dir}!")

# """
# train_model.py
# --------------
# Author: Jennifer Farias Rodrigues
# Date: 10/03/2025
# Description: Train predictive models for IoT logistics optimization.
#
# Structure:
# - Load processed data from MariaDB
# - Split dataset into train (80%) and test (20%)
# - Train multiple models (ARIMA, Random Forest, XGBoost)
# - Save trained models for evaluation
# """
#
# import os
# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from statsmodels.tsa.arima.model import ARIMA
# from scripts.load_data import load_data  # Import processed data
#
# # Load processed data
# df = load_data()
#
# # Ensure data is loaded correctly
# if df is None or df.empty:
#     raise ValueError("Error: No data loaded. Check database connection and data processing.")
#
# # Define target variable and features
# target = "fuel_consumption_rate"
#
# if target not in df.columns:
#     raise ValueError(f"Error: Target variable '{target}' not found in dataset columns: {df.columns}")
# # Trained on Manjaro-oficial training
# features = df.drop(columns=[target])
# # To training for a second time(windows) Drop target and timestamp column-not use colluns on model
# # columns_to_exclude = [
# #     target,
# #     "timestamp",
# #     "risk_classification",         # Categórica (string)
# #     "cargo_condition_status"       # Provavelmente categórica também
# # ]
#
# features = df.drop(columns=columns_to_exclude)
#
#
#
# # Train-test split (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)
#
# # Ensure the models directory exists
# model_dir = "models"
# os.makedirs(model_dir, exist_ok=True)
#
# # Train Random Forest model
# print("Training Random Forest model...")
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Train XGBoost model
# print("Training XGBoost model...")
# xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
# xgb_model.fit(X_train, y_train)
#
# # Save models
# rf_model_path = os.path.join(model_dir, "random_forest.pkl")
# xgb_model_path = os.path.join(model_dir, "xgboost.pkl")
#
# with open(rf_model_path, "wb") as f:
#     pickle.dump(rf_model, f)
#
# with open(xgb_model_path, "wb") as f:
#     pickle.dump(xgb_model, f)
#
# print(f"Models trained and saved successfully in {model_dir}!")
