"""
graphic_analysis.py
--------------------------------------------------
Author: Jennifer Farias Rodrigues
Date: 13/03/2025
Description: Generate bar chart comparing MAE and MAPE for feature ablation scenarios.

This script visualizes the results of the ablation study comparing the effect
of removing key features on fuel consumption prediction using the XGBoost model.
The output is a bar chart comparing MAE and MAPE for each scenario.
"""

import matplotlib.pyplot as plt
import os
os.makedirs("../train_model_ablation/figures/train_model_ablation", exist_ok=True)

# Get the absolute path to the current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory path for saving the figure
output_dir = os.path.join(base_dir, "..", "..", "figures", "ablation")
os.makedirs(output_dir, exist_ok=True)

# Define output image file path
output_path = os.path.join(output_dir, "train_model_ablation_comparison.png")

# Ablation test results
model_names = [
    "Full Model",
    "Without warehouse_inventory_level",
    "Without traffic_congestion_level",
    "Without vehicle_gps_longitude"
]
mae_values = [0.1370, 2.8015, 2.8527, 2.8611]
mape_values = [2.74, 35.47, 36.80, 36.82]

# Create the comparison bar chart
plt.figure(figsize=(10, 6))
plt.barh(model_names, mae_values, height=0.4, label="MAE", color="orange")
plt.barh([name + " " for name in model_names], mape_values, height=0.4, label="MAPE", color="coral", alpha=0.7)

plt.xlabel("Error")
plt.title("MAE and MAPE Error Comparison by Ablation Scenario - XGBoost")
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
# plt.savefig(output_path)
# plt.show()

# Save the figure to the correct local path
# output_path = "figures/train_model_ablation/train_model_ablation_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi 300 for high resolution
plt.show()
