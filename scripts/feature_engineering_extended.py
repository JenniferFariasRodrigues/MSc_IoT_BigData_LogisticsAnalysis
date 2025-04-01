# =====================================================
# Script: feature_engineering_extended.py
# Description: Generates additional derived features to
#              improve model performance in fuel prediction
# Project: MSc_IoT_BigData_LogisticsAnalysis
# Author: Jennifer Farias Rodrigues
# Date: 2025-04-01
# =====================================================

import pandas as pd
import os

# Load original processed data
input_path = 'C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv'
output_path = 'C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/refined_logistics_data.csv'

# Read the CSV file
df = pd.read_csv(input_path)

# ========== NEW DERIVED FEATURES ==========

# 1. Interaction: traffic congestion × warehouse inventory
df["congestion_x_inventory"] = df["traffic_congestion_level"] * df["warehouse_inventory_level"]

# ========== SAVE THE ENHANCED DATASET ==========

os.makedirs('data', exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Feature engineering completed. File saved to: {output_path}")
