"""
train_model_feature_importance_xgboost.py
------------------------------
Author: Jennifer Farias Rodrigues
Date: 17/03/2025
Description: This script analyzes the feature importance of the XGBoost model in fuel consumption prediction.
It specifically investigates whether the "id" column influences the model, which could indicate data leakage
or an unintended relationship.

Structure:
- Load dataset using the `load_data` function.
- Check if the "id" column exists and compute its correlation with `fuel_consumption_rate`.
- Remove "id" if necessary and proceed with feature selection.
- Train an XGBoost model and compute feature importance.
- Compare results before and after removing "id" and visualize the impact.
- Evaluate the model and analyze changes in MAE.

Usage:
Run the script in PyCharm inside the `scripts/` directory:
    python scripts/feature_importance_xgboost.py
"""

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from load_data import load_data  # Importing the function from the project

# Load dataset
df = load_data()

# Check if 'id' exists and compute its correlation with fuel consumption
if 'id' in df.columns:
    correlation_id = df[['id', 'fuel_consumption_rate']].corr().iloc[0, 1]
    print(f"Correlation between ID and Fuel Consumption: {correlation_id:.4f}")

    # If correlation is high, investigate possible issues
    if abs(correlation_id) > 0.5:
        print("Warning: The 'id' column may be influencing predictions. Removing it...")
    df = df.drop(columns=['id'])

# Separate features and target variable
X = df.drop(columns=['fuel_consumption_rate'])
y = df['fuel_consumption_rate']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model_xgb = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model_xgb.fit(X_train_scaled, y_train)

# Compute feature importance
feature_importance = model_xgb.feature_importances_
feature_names = X.columns

# Convert results into a DataFrame and sort by importance
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot feature importance results
plt.figure(figsize=(10, 5))
sns.barplot(y=feature_df['Feature'], x=feature_df['Importance'], palette='Blues_r')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - XGBoost (Without ID)")
plt.savefig("/home/jennifer/Documentos/Dissertação/spark/IoTLogisticsEnergyAnalysis/figures/feature_importance_xgboost_fixed.png")
plt.show()

# Evaluate model performance after removing 'id'
y_pred = model_xgb.predict(X_test_scaled)
mae_xgb = mean_absolute_error(y_test, y_pred)

print(f"MAE of XGBoost without ID: {mae_xgb:.4f}")
