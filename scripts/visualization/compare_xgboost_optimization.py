"""
compare_xgboost_optimization.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 01/04/2025
Description: Generate a bar chart comparing XGBoost models' performance metrics.

This script compares the performance of three XGBoost models:
- Original (baseline)
- Optimized via RandomizedSearchCV (MAE)
- Optimized via GridSearchCV (MAPE)

The evaluation metrics are MAE, RMSE, and MAPE.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# === Model comparison data ===
models = ["XGBoost Original", "RandomSearch (MAE)", "GridSearch (MAPE)"]

mae_values = [0.1370, 0.9109, 3.3041]
rmse_values = [0.1370, 2.1211, 4.1298]
mape_values = [2.74, 12.11, 42.61]

# === Bar settings ===
bar_width = 0.25
x = np.arange(len(models))

# === Create figure ===
plt.figure(figsize=(12, 7))
plt.bar(x - bar_width, mae_values, width=bar_width, label="MAE", color="skyblue")
plt.bar(x, rmse_values, width=bar_width, label="RMSE", color="orange")
plt.bar(x + bar_width, mape_values, width=bar_width, label="MAPE", color="lightcoral")

# === Labels and appearance ===
plt.xlabel("Modelos", fontsize=14)
plt.ylabel("Erro", fontsize=14)
plt.title("Comparação de Erros - XGBoost", fontsize=16)
plt.xticks(x, models, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# === Save figure ===
output_path = "figures/optimization/xgboost_comparative_optimization_final.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Gráfico salvo em: {output_path}")
