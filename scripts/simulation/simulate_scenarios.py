"""
simulate_scenarios.py
---------------------
Author: Jennifer Farias Rodrigues
Date: 20/05/2025
Description: This script generates synthetic samples of key operational variables
for different simulation scenarios (Ideal, Realistic, Optimized), using predefined
configuration ranges.

Structure:
- Import configuration dictionary from scenario_config.py
- Use NumPy to generate random samples for three variables:
  - handling_equipment_efficiency
  - warehouse_inventory_level
  - traffic_congestion_level
- Return the result as a labeled Pandas DataFrame for further evaluation

==>What will happen when running:
5 simulated samples will be generated with three numeric columns:handling_equipment_efficiency

warehouse_inventory_level, traffic_congestion_level

An additional column called scenario will be added with the name of the scenario.

The DataFrame will be displayed in the terminal.

Usage:
python scripts/simulation/simulate_scenarios.py
"""

import numpy as np
import pandas as pd
from scripts.simulation.scenario_config import SCENARIO_CONFIG

def generate_simulation_samples(scenario_name: str, n_samples: int = 100) -> pd.DataFrame:
    """
    Generate simulated samples of operational variables for the specified scenario.

    Parameters:
        scenario_name (str): scenario name (e.g., 'ideal', 'realistic', 'optimized')
        n_samples (int): number of samples to generate

    Returns:
        pd.DataFrame: a DataFrame with n_samples rows containing the simulated values
    """
    if scenario_name not in SCENARIO_CONFIG:
        raise ValueError(f"Invalid scenario: {scenario_name}. Choose from: {list(SCENARIO_CONFIG.keys())}")

    config = SCENARIO_CONFIG[scenario_name]
    np.random.seed(42)

    data = {
        'handling_equipment_efficiency': np.round(np.random.uniform(*config['handling_equipment_efficiency'], n_samples), 2),
        'warehouse_inventory_level': np.round(np.random.uniform(*config['warehouse_inventory_level'], n_samples), 2),
        'traffic_congestion_level': np.round(np.random.uniform(*config['traffic_congestion_level'], n_samples), 2)
    }

    df = pd.DataFrame(data)
    df['scenario'] = scenario_name
    return df

# Manual test
if __name__ == "__main__":
    df_simulated = generate_simulation_samples('ideal', 5)
    print(df_simulated)
