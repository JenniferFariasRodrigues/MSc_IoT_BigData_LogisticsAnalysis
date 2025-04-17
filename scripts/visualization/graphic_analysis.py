# =====================================================
# Script: graphic_analysis_ablation.py
# Description: Generate MAE and MAPE bar charts for ablation scenarios
# Project: MSc_IoT_BigData_LogisticsAnalysis
# Author: Jennifer Farias Rodrigues
# Date: 2025-04-13
# =====================================================

import matplotlib.pyplot as plt
import os

# === Setup directory ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "..", "..", "figures", "ablation")
os.makedirs(output_dir, exist_ok=True)

# === Data ===
model_names = [
    "Full Model",
    "Without warehouse_inventory_level",
    "Without traffic_congestion_level",
    "Without vehicle_gps_longitude"
]
mae_values = [0.1370, 2.8015, 2.8527, 2.8611]
mape_values = [2.74, 35.47, 36.80, 36.82]

# === MAE Chart ===
fig, ax = plt.subplots(figsize=(10, 5.5))  # mais espaço horizontal
bars_mae = ax.barh(model_names, mae_values, color="#f4a300")

ax.set_title("MAE by Ablation Scenario", fontsize=22, pad=15)
ax.set_xlabel("MAE", fontsize=20)
ax.set_ylabel("Model Scenario", fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.grid(axis='x', linestyle='--', alpha=0.5)

max_mae = max(mae_values)
for bar, value in zip(bars_mae, mae_values):
    ax.text(value + 0.1, bar.get_y() + bar.get_height()/2,
            f"{value:.2f}", va='center', ha='left', fontsize=18)

ax.set_xlim(0, max_mae + 1)
plt.tight_layout()
output_mae = os.path.join(output_dir, "train_model_ablation_mae.png")
plt.savefig(output_mae, dpi=300, bbox_inches='tight')
plt.show()

# === MAPE Chart ===
fig, ax = plt.subplots(figsize=(14, 5.5))
bars_mape = ax.barh(model_names, mape_values, color="#f8b88b")

ax.set_title("MAPE by Ablation Scenario", fontsize=22, pad=15)
ax.set_xlabel("MAPE (%)", fontsize=20)
ax.set_ylabel("Model Scenario", fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.grid(axis='x', linestyle='--', alpha=0.5)

max_mape = max(mape_values)
for bar, value in zip(bars_mape, mape_values):
    ax.text(value + 0.5, bar.get_y() + bar.get_height()/2,
            f"{value:.2f}", va='center', ha='left', fontsize=18)

ax.set_xlim(0, max_mape + 4)
plt.tight_layout()
output_mape = os.path.join(output_dir, "train_model_ablation_mape.png")
plt.savefig(output_mape, dpi=300, bbox_inches='tight')
plt.show()

print("✅ Fontes ampliadas com sucesso!")
print(f"   • MAE:  {output_mae}")
print(f"   • MAPE: {output_mape}")



# """
# graphic_analysis.py
# --------------------------------------------------
# Author: Jennifer Farias Rodrigues
# Date: 13/03/2025
# Description: Generate bar chart comparing MAE and MAPE for feature ablation scenarios.
#
# This script visualizes the results of the ablation study comparing the effect
# of removing key features on fuel consumption prediction using the XGBoost model.
# The output is a bar chart comparing MAE and MAPE for each scenario.
# """
#
# import matplotlib.pyplot as plt
# import os
# os.makedirs("../train_model_ablation/figures/train_model_ablation", exist_ok=True)
#
# # Get the absolute path to the current script directory
# base_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Create output directory path for saving the figure
# output_dir = os.path.join(base_dir, "..", "..", "figures", "ablation")
# os.makedirs(output_dir, exist_ok=True)
#
# # Define output image file path
# output_path = os.path.join(output_dir, "train_model_ablation_comparison.png")
#
# # Ablation test results
# model_names = [
#     "Full Model",
#     "Without warehouse_inventory_level",
#     "Without traffic_congestion_level",
#     "Without vehicle_gps_longitude"
# ]
# mae_values = [0.1370, 2.8015, 2.8527, 2.8611]
# mape_values = [2.74, 35.47, 36.80, 36.82]
#
# # Create the comparison bar chart
# plt.figure(figsize=(10, 6))
# plt.barh(model_names, mae_values, height=0.4, label="MAE", color="orange")
# plt.barh([name + " " for name in model_names], mape_values, height=0.4, label="MAPE", color="coral", alpha=0.7)
#
# plt.xlabel("Error")
# plt.title("MAE and MAPE Error Comparison by Ablation Scenario - XGBoost")
# plt.legend()
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
#
# # Save the figure
# # plt.savefig(output_path)
# # plt.show()
#
# # Save the figure to the correct local path
# # output_path = "figures/train_model_ablation/train_model_ablation_comparison.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi 300 for high resolution
# plt.show()
