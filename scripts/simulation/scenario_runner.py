"""
scenario_runner.py
-------------------
Author: Jennifer Farias Rodrigues
Date: 20/05/2025
Description:
This script runs the full pipeline for scenario-based simulation and evaluation of fuel consumption
in logistics operations. It integrates the modules developed for simulation generation and statistical analysis.

Specifically, this script will:
- Call the function `generate_simulation_samples(...)` from `simulate_scenarios.py` for each scenario
  ('ideal', 'realista', 'otimizado') located in `scripts/simulation/simulate_scenarios.py`
- Concatenate all the generated DataFrames into a single dataset
- Call the function `evaluate_simulated_data(...)` from `evaluate_scenarios.py` located in
  `scripts/simulation/evaluate_scenarios.py` to compute:
    - Mean fuel consumption
    - Standard deviation (variability)
    - Relative consumption reduction percentage compared to the baseline scenario
- Save the simulated samples from each scenario as separate CSV files in `scripts/simulation/results/`
- Export the evaluation summary as `evaluation_summary.csv` to the same directory

Structure:
- Generate data using simulate_scenarios.py
- Evaluate metrics using evaluate_scenarios.py
- Save all outputs in results/

Usage:
python scripts/simulation/scenario_runner.py
"""

import pandas as pd
from scripts.simulation.simulate_scenarios import generate_simulation_samples
from scripts.simulation.evaluate_scenarios import evaluate_simulated_data
import os

# Output directory for results
# output_dir = "scripts/simulation/results"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# List of scenario names
scenarios = ['ideal', 'realista', 'otimizado']
all_data = []

# Generate and save samples for each scenario
for scenario in scenarios:
    df = generate_simulation_samples(scenario_name=scenario, n_samples=100)
    df.to_csv(os.path.join(output_dir, f"{scenario}.csv"), index=False)
    all_data.append(df)

# Combine all scenario samples into a single DataFrame
df_all = pd.concat(all_data, ignore_index=True)

# Evaluate metrics using placeholder fuel consumption formula
evaluation_summary = evaluate_simulated_data(df_all)

# Save summary metrics
evaluation_summary.to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False)

# Display summary in terminal
print(evaluation_summary)