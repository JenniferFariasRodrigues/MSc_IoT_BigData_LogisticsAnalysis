"""
evaluate_scenarios.py
----------------------
Author: Jennifer Farias Rodrigues
Date: 20/05/2025
Description: This script evaluates the simulated samples for each logistics scenario
by computing statistical and operational metrics related to fuel consumption.

The evaluation includes:
- Estimation of fuel consumption using a placeholder formula based on simulated variables
  (to be replaced by the predictive model in future stages)
- Generation of 100 samples for each defined scenario using generate_simulation_samples(...)
- Integration with scenario_config.py for range definitions
- Calculation of mean fuel consumption per scenario
- Standard deviation to assess variability within each scenario
- Percentage reduction in fuel usage compared to the baseline ('ideal') scenario
- Optional metrics such as median or confidence intervals can be added

Structure:
- Load or simulate data for the three scenarios: 'ideal', 'realista', 'otimizado'
- Apply a simplified fuel consumption formula
- Group results by scenario and compute summary statistics
- Output results to terminal for analysis or export

Usage:
python scripts/simulation/evaluate_scenarios.py
"""

import pandas as pd
from scripts.simulation.simulate_scenarios import generate_simulation_samples

def evaluate_simulated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate fuel consumption and summary metrics from simulated scenario data.

    Parameters:
        df (DataFrame): simulated samples with columns for operational variables

    Returns:
        DataFrame: metrics grouped by scenario
    """
    # Placeholder formula — adjust later with actual model
    df['fuel_consumption'] = (
        10
        - 3 * df['handling_equipment_efficiency']
        + 2 * df['traffic_congestion_level']
        - 1.5 * df['warehouse_inventory_level']
    )

    summary = df.groupby('scenario')['fuel_consumption'].agg(['mean', 'std']).reset_index()

    # Optional: add relative reduction to ideal as baseline
    ideal_mean = summary.loc[summary['scenario'] == 'ideal', 'mean'].values[0]
    summary['reduction_percent'] = (ideal_mean - summary['mean']) / ideal_mean * 100

    return summary


if __name__ == "__main__":
    from scripts.simulation.simulate_scenarios import generate_simulation_samples

    # Gerar dados simulados para os três cenários
    df_ideal = generate_simulation_samples('ideal', 100)
    df_realista = generate_simulation_samples('realista', 100)
    df_otimizado = generate_simulation_samples('otimizado', 100)

    # Concatenar todos
    df_all = pd.concat([df_ideal, df_realista, df_otimizado], ignore_index=True)

    # Avaliar
    result = evaluate_simulated_data(df_all)
    print(result)
