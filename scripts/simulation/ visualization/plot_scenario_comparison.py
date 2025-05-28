
"""
plot_scenario_comparison.py
---------------------------

This script loads the summary results of fuel consumption simulations
for three operational scenarios (ideal, optimized, realistic),
and generates a comparative bar chart with annotated labels.

Input:
    - CSV file: simulation/results/evaluation_summary.csv
      Columns: scenario, mean, std, reduction_percent

Output:
    - Bar chart saved to: simulation/visualization/scenario_comparison.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define path to CSV summary file
CSV_PATH = os.path.join("..", "results", "evaluation_summary.csv")

# Load data
df = pd.read_csv(CSV_PATH)

# Define color palette aligned with the article’s style
colors = ["#66BB6A", "#FFA726", "#EF5350"]  # green, orange, red soft tones

# Create clean bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(
    df["scenario"],
    df["mean"],
    color=colors
)

# Adjust Y-axis to avoid label clipping
max_value = df["mean"].max()
plt.ylim(0, max_value + 1.0)

# Add labels and title
plt.title("Fuel Consumption Comparison Across Scenarios", fontsize=12)
plt.ylabel("Fuel Consumption (L/delivery)", fontsize=10)
plt.xlabel("Operational Scenario", fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Annotate each bar with mean and reduction
for bar, mean, reduction in zip(bars, df["mean"], df["reduction_percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{mean:.2f} L\n({reduction:.1f}%)",
        ha="center",
        fontsize=9
    )

# Save figure to the same directory as the script
output_path = os.path.join(os.path.dirname(__file__), "scenario_comparison.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.close()
