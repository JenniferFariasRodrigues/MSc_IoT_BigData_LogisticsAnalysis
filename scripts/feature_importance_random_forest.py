import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.preprocessing.load_data import load_data

# Path to model and target
MODEL_PATH = "models/random_forest.pkl"
TARGET_VARIABLE = "fuel_consumption_rate"

# Load data and model
df = load_data()
if TARGET_VARIABLE not in df.columns:
    raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset.")
X = df.drop(columns=[TARGET_VARIABLE])
with open(MODEL_PATH, "rb") as file:
    rf_model = pickle.load(file)

# Feature importance
importances = rf_model.feature_importances_
feature_names = rf_model.feature_names_in_ if hasattr(rf_model, "feature_names_in_") else X.columns
df_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# === PLOT SETTINGS ===
plt.figure(figsize=(8, 6))  # Menor largura, mantendo boa altura
sns.barplot(data=df_importance, x="Importance", y="Feature", palette="Blues_r", hue="Feature", legend=False)

plt.title("Feature Importance - Random Forest", fontsize=22, fontweight='normal')
plt.xlabel("Importance Score", fontsize=20)
plt.ylabel("Feature", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.15)

# Save figure on C:\Users\Jennifer\Dissertacao\MSc_IoT_BigData_LogisticsAnalysis\scripts\figures\pic_feature_importance
output_path = "figures/pic_feature_importance/feature_importance_random_forest_2025.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print("✅ Random Forest importance plot saved successfully.")

# """
# feature_importance_random_forest.py
# -----------------------------------
# Author: Jennifer Farias Rodrigues
# Date: 17/03/2025
# Description: Generates a feature importance plot for the trained Random Forest model
# with increased font size for publication purposes.
#
# Steps:
# - Load data using `load_data()`
# - Load trained Random Forest model from pickle
# - Compute feature importance
# - Plot bar chart with larger fonts
# - Save the figure in high resolution
# """
#
# import os
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scripts.preprocessing.load_data import load_data
#
# # Path to the saved Random Forest model
# MODEL_PATH = "models/random_forest.pkl"
# TARGET_VARIABLE = "fuel_consumption_rate"
#
# # Step 1: Load dataset
# print("Loading dataset...")
# df = load_data()
#
# if TARGET_VARIABLE not in df.columns:
#     raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in the dataset.")
#
# # Separate features and target
# X = df.drop(columns=[TARGET_VARIABLE])
#
# # Step 2: Load trained model
# with open(MODEL_PATH, "rb") as file:
#     rf_model = pickle.load(file)
#
# # Step 3: Get feature names
# if hasattr(rf_model, "feature_names_in_"):
#     feature_names = rf_model.feature_names_in_
# else:
#     feature_names = X.columns
#
# # Step 4: Compute feature importances
# importances = rf_model.feature_importances_
# df_importance = pd.DataFrame({
#     "Feature": feature_names,
#     "Importance": importances
# }).sort_values(by="Importance", ascending=False)
#
# # Step 5: Plot with increased font size
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df_importance, x="Importance", y="Feature", palette="Blues_r", hue="Feature", legend=False)
#
# # Aumentar tamanho das fontes e deixar tudo coerente com o artigo
# plt.title("Feature Importance - Random Forest", fontsize=28, fontweight='normal')  # sem negrito
# plt.xlabel("Importance Score", fontsize=24)
# plt.ylabel("Feature", fontsize=24)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#
# # Ajustar espaçamento para não cortar nomes longos à esquerda
# # plt.subplots_adjust(left=0.50, right=0.90, top=0.9, bottom=0.2)
# plt.subplots_adjust(left=0.45, right=0.85, top=0.9, bottom=0.2)
#
#
#
# # Step 6: Save figure to file on C:\Users\Jennifer\Dissertacao\MSc_IoT_BigData_LogisticsAnalysis\scripts\figures\pic_feature_importance
# os.makedirs("figures/pic_feature_importance", exist_ok=True)
# plt.savefig("figures/pic_feature_importance/feature_importance_random_forest_2025_III.png", dpi=300)
# plt.show()
#
# print("Feature importance plot saved successfully.")
