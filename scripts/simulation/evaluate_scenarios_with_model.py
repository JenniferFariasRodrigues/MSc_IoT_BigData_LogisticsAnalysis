"""
evaluate_scenarios_with_model.py
--------------------------------
Author: Jennifer Farias Rodrigues
Date: 23/05/2025
Description:
This script evaluates fuel consumption in simulated scenarios using a pre-trained XGBoost model.

Instead of a placeholder formula, it loads the real model trained on IoT logistics data and applies it
to synthetic scenario samples generated previously.

Structure:
- Load simulation samples from /results/*.csv
- Load the XGBoost model from /models/xgboost_model.pkl
- Apply the model to predict fuel consumption per sample
- Compute summary statistics (mean, std, reduction%)
- Export evaluated results to new CSV files in /results/evaluated/

Usage:
python scripts/simulation/evaluate_scenarios_with_model.py
"""

import os
import glob
import pandas as pd
from joblib import load

RESULTS_DIR = "results"
EVALUATED_DIR = os.path.join(RESULTS_DIR, "evaluated")

os.makedirs(EVALUATED_DIR, exist_ok=True)

# Load trained model
model = load("models/xgboost.pkl")

# Identify scenario CSVs
scenario_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
summary_list = []

for file_path in scenario_files:
    df = pd.read_csv(file_path)
    scenario_name = df["scenario"].iloc[0]

    # Select only the columns expected by the model
    X = df[["handling_equipment_efficiency", "warehouse_inventory_level", "traffic_congestion_level"]]

    # Predict fuel consumption using real model
    df["fuel_consumption"] = model.predict(X)

    # Save individual evaluated scenario file
    evaluated_file = os.path.join(EVALUATED_DIR, f"{scenario_name}_evaluated.csv")
    df.to_csv(evaluated_file, index=False)

    # Aggregate statistics
    mean = df["fuel_consumption"].mean()
    std = df["fuel_consumption"].std()
    summary_list.append({
        "scenario": scenario_name,
        "mean_fuel_consumption": round(mean, 3),
        "std_fuel_consumption": round(std, 3)
    })

# Create summary table
summary_df = pd.DataFrame(summary_list)

# Calculate percentage reduction using "ideal" as reference
ideal_mean = summary_df.loc[summary_df["scenario"] == "ideal", "mean_fuel_consumption"].values[0]
summary_df["reduction_percent"] = ((ideal_mean - summary_df["mean_fuel_consumption"]) / ideal_mean * 100).round(2)

# Save summary
summary_df.to_csv(os.path.join(EVALUATED_DIR, "scenario_evaluation_summary.csv"), index=False)

print("\n✅ Evaluation complete. Results saved to:", EVALUATED_DIR)
print(summary_df)
