# =====================================================
# Script: plot_correlation_matrix_key_variables.py
# Description: Generates a Pearson correlation matrix
#              for key IoT features using data from MariaDB.
#              Saves figure with large axis labels for publication.
# Project: MSc_IoT_BigData_LogisticsAnalysis
# Author: Jennifer Farias Rodrigues
# Date: 2025-04-24
# =====================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load processed data
data= pd.read_csv("C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv")

# Manually rename columns to include line breaks
renamed_columns = {
    "fuel_consumption_rate": "fuel\nconsumption\nrate",
    "warehouse_inventory_level": "warehouse\ninventory\nlevel",
    "traffic_congestion_level": "traffic\ncongestion\nlevel",
    "handling_equipment_efficiency": "handling\nequipment\nefficiency"
}
data = data.rename(columns=renamed_columns)

# Select only the top important features
selected_features = list(renamed_columns.values())

# Compute Pearson correlation matrix
correlation_matrix = data[selected_features].corr(method="pearson")

# Plot settings
sns.set(style="white", font_scale=1.8)
plt.figure(figsize=(16, 10))

# Generate heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={"size": 26},
    cbar_kws={"shrink": 0.8}
)

plt.title("Pearson Correlation Matrix of Key IoT Variables", fontsize=32, weight='bold')

# Apply font size and alignment for axis labels
plt.xticks(
    rotation=0,
    fontsize=28,
    ha='center',
weight='bold'
)
plt.yticks(
    rotation=0,
    fontsize=30,
weight='bold'
)

plt.subplots_adjust(bottom=0.30, left=0.30, right=0.95, top=0.93)


# Save figure to file
plt.savefig("figures/correlation/key_variables_correlation_matrix.png", dpi=300)
plt.show()

# # =====================================================
# # Script: plot_correlation_matrix_key_variables.py
# # Description: Generates a Pearson correlation matrix
# #              for key IoT features using data from MariaDB.
# #              Saves figure with large axis labels for publication.
# # Project: MSc_IoT_BigData_LogisticsAnalysis
# # Author: Jennifer Farias Rodrigues
# # Date: 2025-04-24
# # =====================================================
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scripts.preprocessing.load_data import load_data
#
# # Load processed data from MariaDB
# # data = load_data()
# data= pd.read_csv("C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv")
# # Select only the top important features
# selected_features = [
#     "fuel_consumption_rate",
#     "warehouse_inventory_level",
#     "traffic_congestion_level",
#     "handling_equipment_efficiency"
# ]
#
# # Compute Pearson correlation matrix
# correlation_matrix = data[selected_features].corr(method="pearson")
#
# # Plot settings
# sns.set(style="white", font_scale=1.6)  # Increase font scale for all elements
# plt.figure(figsize=(11, 9))
#
# # Generate heatmap
# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1,
#     linewidths=0.5,
#     annot_kws={"size": 24},  # Font size inside cells
#     cbar_kws={"shrink": 0.8}
# )
#
# plt.title("Pearson Correlation Matrix of Key IoT Variables", fontsize=24, weight='bold')
# plt.xticks(rotation=45, ha='right', fontsize=22)
# plt.yticks(rotation=0, fontsize=22)
# plt.tight_layout()
#
# # Adjust layout manually to prevent cutoffs
# # plt.subplots_adjust(bottom=0.30, left=0.35, right=0.98, top=0.92)
# # plt.figure(figsize=(12, 10))
# plt.subplots_adjust(bottom=0.37, left=0.40, right=1.0, top=0.94)
#
#
# # Save figure to file
# plt.savefig("figures/correlation/key_variables_correlation_matrix_I.png", dpi=300)
# plt.show()

# Old code II
# # =====================================================
# # Script: plot_correlation_matrix_key_variables.py
# # Description: Generates a Pearson correlation matrix
# #              for key IoT features using data from MariaDB.
# #              Saves figure with large axis labels for publication.
# # Project: MSc_IoT_BigData_LogisticsAnalysis
# # Author: Jennifer Farias Rodrigues
# # Date: 2025-04-24
# # =====================================================
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scripts.preprocessing.load_data import load_data
#
# # Load dataset from CSV
# data = pd.read_csv("C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv")
# # data = load_data()
# # Select only the top important features
# selected_features = [
#     "fuel_consumption_rate",
#     "warehouse_inventory_level",
#     "traffic_congestion_level",
#     "handling_equipment_efficiency"
# ]
#
# # Compute correlation matrix
# correlation_matrix = data[selected_features].corr(method="pearson")
#
#
# sns.set(style="white", font_scale=1.4)
# plt.figure(figsize=(8, 6))
#
# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1,
#     linewidths=0.5,
#     annot_kws={"size": 13},
#     cbar_kws={"shrink": 0.8}
# )
#
# plt.title("Pearson Correlation Matrix of Key IoT Variables", fontsize=16, weight='bold')
# plt.xticks(rotation=45, ha='right', fontsize=13)
# plt.yticks(rotation=0, fontsize=13)
# plt.tight_layout()
#
# # Save figure to file
# plt.savefig("figures/correlation/key_variables_correlation_matrix.png", dpi=300)
# plt.show()


# """
#Old code.
# """
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Load dataset from CSV
# data = pd.read_csv("C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv")
#
# # Select only the top important features
# selected_features = [
#     "fuel_consumption_rate",
#     "warehouse_inventory_level",
#     "traffic_congestion_level",
#     "handling_equipment_efficiency"
# ]
#
# # Compute correlation matrix
# correlation_matrix = data[selected_features].corr(method="pearson")
#
# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Pearson Correlation Matrix of Key IoT Variables")
# plt.tight_layout()
#
# # Save figure to file
# plt.savefig("figures/correlation/key_variables_correlation_matrix.png", dpi=300)
# plt.show()
