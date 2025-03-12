"""
evaluate_models.py
------------------
Author: Jennifer Farias Rodrigues
Date: 11/03/2025
Description: Evaluate trained predictive models for IoT logistics optimization.

This script loads the trained machine learning models (Random Forest, XGBoost),
performs predictions on the test dataset, and evaluates their performance using
standard error metrics.

Structure:
- Load processed data from MariaDB.
- Split dataset into train (80%) and test (20%).
- Load trained models (Random Forest, XGBoost).
- Evaluate models using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
- Save results in CSV format for analysis.


"""

import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.load_data import load_data  # Import processed data
import numpy as np

# Load processed data
df = load_data()

# Check if data was loaded correctly
if df is None or df.empty:
    print("Error: No data loaded. Check the database connection or query.")
    exit(1)

# Display the first rows and available columns
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# Verify if 'fuel_consumption_rate' exists
target_column = "fuel_consumption_rate"
if target_column not in df.columns:
    print(f"Error: Column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    exit(1)

# Separate features and target variable
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data into training (80%) and test (20%) sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained models
try:
    with open("models/random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("models/xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the models are trained and saved in 'models' directory.")
    exit(1)

# Make predictions on test set
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (abs(y_true - y_pred) / y_true).mean() * 100  # MAPE in %
    return mae, rmse, mape

# Evaluate Random Forest
mae_rf, rmse_rf, mape_rf = calculate_metrics(y_test, y_pred_rf)

# Evaluate XGBoost
mae_xgb, rmse_xgb, mape_xgb = calculate_metrics(y_test, y_pred_xgb)

# Create results table
results_df = pd.DataFrame({
    "Modelo": ["Random Forest", "XGBoost"],
    "MAE": [mae_rf, mae_xgb],
    "RMSE": [rmse_rf, rmse_xgb],
    "MAPE (%)": [mape_rf, mape_xgb]
})

# Display results
print("\nEvaluation Results:")
print(results_df)

# Save results table
results_df.to_csv("models/evaluation_results.csv", index=False)
print("\nResults saved in 'models/evaluation_results.csv'")
