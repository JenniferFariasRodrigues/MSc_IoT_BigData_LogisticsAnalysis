# scripts/simulation/scenario_config.py
"""
scenario_config.py
-------------------
Author: Jennifer Farias Rodrigues
Date: 20/05/2025
Description:
This file defines the configuration ranges for the three main scenarios used in the simulation:
- Ideal
- Realistic
- Inventory-optimized

The parameter ranges were defined based on:
- Empirical values derived from the exploratory analysis of IoT-based logistics data
- Percentiles (min, median, max) of the actual dataset (normalized between 0 and 1)
- Literature references that support the expected ranges for each variable:
    * Handling Equipment Efficiency: based on fuel-efficiency modeling from Moradi & Miranda-Moreno (2020)
    * Warehouse Inventory Level: informed by optimal inventory thresholds (Fan et al., 2024)
    * Traffic Congestion Level: scenario-based simulations on urban congestion (Mashhadi et al., 2025; Yu et al., 2024)

Each scenario reflects a combination of these operational settings and supports simulation-based evaluation of fuel consumption.
"""

SCENARIO_CONFIG = {
    'ideal': {
        'handling_equipment_efficiency': (0.95, 1.00),
        'warehouse_inventory_level': (0.48, 0.52),
        'traffic_congestion_level': (0.05, 0.15)
    },
    'realista': {
        'handling_equipment_efficiency': (0.50, 0.70),
        'warehouse_inventory_level': (0.40, 0.70),
        'traffic_congestion_level': (0.50, 0.80)
    },
    'otimizado': {
        'handling_equipment_efficiency': (0.60, 0.80),
        'warehouse_inventory_level': (0.70, 0.90),
        'traffic_congestion_level': (0.40, 0.60)
    }
}
