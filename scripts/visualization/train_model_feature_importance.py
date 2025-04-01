# =====================================================
# Script: train_model_feature_importance.py
# Description: Trains XGBoost model and plots feature
#              importance using the refined dataset
# Project: MSc_IoT_BigData_LogisticsAnalysis
# Author: Jennifer Farias Rodrigues
# Date: 2025-04-01
# =====================================================

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# Load dataset with engineered features
input_path = 'C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/refined_logistics_data.csv'
df = pd.read_csv(input_path)

# Define target and features
target = 'fuel_consumption_rate'

# Drop identifier columns that shouldn't be used for training
columns_to_drop = ['id', 'vehicle_id', 'rota_id']
X = df.drop(columns=[target] + [col for col in columns_to_drop if col in df.columns])
y = df[target]
print("\nVariáveis de entrada (features):")
for col in X.columns:
    print(f" - {col}")

# Train XGBoost regressor
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Plot feature importance
# fig, ax = plt.subplots(figsize=(12, max(4, len(X.columns) * 0.01)))  # altura dinâmica
# xgb.plot_importance(model, importance_type='gain', height=0.5, ax=ax)
# ax.set_title('XGBoost Feature Importance (Refined Dataset)')
# fig.tight_layout(pad=1)

# fig, ax = plt.subplots(figsize=(14, 6))  # Largura equilibrada, altura reduzida
# xgb.plot_importance(model, importance_type='gain', height=0.5, ax=ax)
#
# ax.set_title('XGBoost Feature Importance (Refined Dataset)', fontsize=14, pad=12)
# ax.set_xlabel('Importance score', fontsize=12)
# ax.set_ylabel('Features', fontsize=12)
#
# Ajusta melhor margens e espaço para rótulos
# fig.tight_layout(pad=2, rect=[0.2, 0.1, 0.98, 0.95])

fig, ax = plt.subplots(figsize=(10, 6))  # largura fixa, altura dinâmica
xgb.plot_importance(model, importance_type='gain', height=0.5, ax=ax)

ax.set_title('XGBoost Feature Importance (Refined Dataset)', fontsize=14, pad=12)
ax.set_xlabel('Importance score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)


# Save figure
output_dir = 'figures/pic_feature_importance'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'feature_importance_xgboost_extended_II.png')
fig.savefig(output_file, bbox_inches='tight')

# plt.savefig(output_file)

print(f" Feature importance plot saved to: {output_file}")
