"""
refine_xgboost.py
------------------------
Author: Jennifer Farias Rodrigues
Date: 17/03/2025
Description: This script refines the XGBoost model by selecting only the most relevant features based on previous feature importance analysis.

Process:
- Load the dataset from MariaDB (using load_data.py)
- Select key features: traffic_congestion_level, vehicle_gps_longitude, warehouse_inventory_level
- Train an optimized XGBoost model
- Evaluate performance before and after feature selection
- Compare MAE, RMSE, and MAPE to check improvements

Usage:
python scripts/refine_xgboost.py
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.load_data import load_data  # Importing the function to fetch data from MariaDB

# Load dataset
df = load_data()

# Ensure dataset is loaded correctly
if df is None or df.empty:
    print("Error: No data loaded. Check the database connection.")
    exit()

# Define the target variable
target = 'fuel_consumption_rate'

# Step 1: Keep only relevant features based on feature importance analysis
selected_features = [
    "traffic_congestion_level",
    "vehicle_gps_longitude",
    "warehouse_inventory_level"
]

# Step 2: Filter dataset to only include selected features
df_selected = df[selected_features + [target]]

# Step 3: Split dataset into training and testing sets
X = df_selected.drop(columns=[target])
y = df_selected[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train optimized XGBoost model
model_xgb = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model_xgb.fit(X_train_scaled, y_train)

# Step 6: Evaluate model performance
y_pred = model_xgb.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Display results
print("\n=== XGBoost Performance After Feature Selection ===")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Step 7: Plot Feature Importance
feature_importance = model_xgb.feature_importances_
feature_names = X.columns

feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(y=feature_df['Feature'], x=feature_df['Importance'], palette='Blues_r')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - XGBoost (Refined)")
plt.savefig("/home/jennifer/Documentos/Dissertação/spark/IoTLogisticsEnergyAnalysis/figures/feature_importance_xgboost_refined.png")  # Save refined figure
plt.show()
