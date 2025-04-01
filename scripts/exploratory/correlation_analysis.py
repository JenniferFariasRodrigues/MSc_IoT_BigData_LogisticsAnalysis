"""
correlation_analysis.py

This script performs a Pearson correlation analysis on the processed IoT logistics dataset.
It generates a correlation matrix heatmap to visualize the linear relationships between numerical features,
supporting exploratory data analysis (EDA) and feature selection.

Author: Jennifer Farias Rodrigues
Date: April 2025
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define the path to the processed dataset
data_path = "C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv"

# Define the output directory and create it if it doesn't exist
output_dir = "figures/correlation"
os.makedirs(output_dir, exist_ok=True)

# Set the output image filename
output_path = os.path.join(output_dir, "correlation_matrix.png")

# Load the dataset
df = pd.read_csv(data_path)

# Drop non-numerical or irrelevant columns from correlation analysis
columns_to_exclude = ['id', 'timestamp']  # Adjust depending on the dataset structure
df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

# Calculate Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Pearson Correlation Matrix of IoT Logistics Variables")
plt.tight_layout()

# Save the figure
plt.savefig(output_path)
plt.close()

print(f"Correlation matrix saved to: {output_path}")
