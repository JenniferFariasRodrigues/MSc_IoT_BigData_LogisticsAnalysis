# """
# Script to generate a Pearson correlation matrix for selected key features.
# This is useful for highlighting the relationship between top important variables.
# """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset from CSV
data = pd.read_csv("C:/Users/Jennifer/Dissertacao/MSc_IoT_BigData_LogisticsAnalysis/data/processed_logistics_data.csv")

# Select only the top important features
selected_features = [
    "fuel_consumption_rate",
    "warehouse_inventory_level",
    "traffic_congestion_level",
    "handling_equipment_efficiency"
]

# Compute correlation matrix
correlation_matrix = data[selected_features].corr(method="pearson")


sns.set(style="white", font_scale=1.4)
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={"size": 13},
    cbar_kws={"shrink": 0.8}
)

plt.title("Pearson Correlation Matrix of Key IoT Variables", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(rotation=0, fontsize=13)
plt.tight_layout()

# Save figure to file
plt.savefig("figures/correlation/key_variables_correlation_matrix.png", dpi=300)
plt.show()


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
