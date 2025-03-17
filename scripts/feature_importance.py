"""
feature_importance.py
------------------------
Author: Jennifer Farias Rodrigues
Date: 05/03/2025
Description: This script computes feature importance for fuel consumption prediction models.

Structure:
- Load data using the `load_data` function from `load_data.py`
- Load trained models from the `models/` directory
- Compute feature importance for:
  - **Random Forest**
  - **XGBoost**
- Visualize feature importance using bar plots
- Save feature importance results to CSV files in `results/`

Usage:
python scripts/feature_importance.py
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.load_data import load_data  # Importing the function to load data

# Directory where trained models are stored
MODEL_DIR = "models/"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")

# Target variable for prediction
TARGET_VARIABLE = "fuel_consumption_rate"

# Step 1: Load data using the function from load_data.py
print("Loading dataset from MariaDB...")
df_pandas = load_data()

if df_pandas is None:
    raise ValueError("No data was loaded. Please check the database connection.")

# Ensure the target variable exists in the dataset
if TARGET_VARIABLE not in df_pandas.columns:
    raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset.")

# Separate features and target variable
X = df_pandas.drop(columns=[TARGET_VARIABLE])
feature_names = X.columns.tolist()

# Function to load a trained model from a pickle file
def load_model(model_path):
    """Loads a trained machine learning model from a pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Step 2: Load trained models
print("Loading trained models...")
rf_model = load_model(RF_MODEL_PATH)
xgb_model = load_model(XGB_MODEL_PATH)

# Step 3: Compute feature importance for both models
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# Create DataFrames for visualization
df_rf = pd.DataFrame({"Feature": feature_names, "Importance": rf_importance})
df_xgb = pd.DataFrame({"Feature": feature_names, "Importance": xgb_importance})

# Sort features by importance in descending order
df_rf = df_rf.sort_values(by="Importance", ascending=False)
df_xgb = df_xgb.sort_values(by="Importance", ascending=False)

# Step 4: Plot feature importance for Random Forest
plt.figure(figsize=(12, 6))
sns.barplot(x=df_rf["Importance"], y=df_rf["Feature"])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Step 5: Plot feature importance for XGBoost
plt.figure(figsize=(12, 6))
sns.barplot(x=df_xgb["Importance"], y=df_xgb["Feature"])
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Step 6: Save feature importance results as CSV files
os.makedirs("results", exist_ok=True)
df_rf.to_csv("results/feature_importance_rf.csv", index=False)
df_xgb.to_csv("results/feature_importance_xgb.csv", index=False)

print("Feature importance analysis completed. Results saved successfully.")
