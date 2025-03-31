"""
feature_engineering.py
------------------------
Author: Jennifer Farias Rodrigues
Date: 05/03/2025
Description: This script performs feature engineering on logistics data and saves the processed data back to MariaDB.

Structure:
- Load raw data from MariaDB using `load_data.py`
- Normalize numeric features using Min-Max Scaling
- Create new feature: Traffic Density
- Save the processed data back to MariaDB

Usage:
python scripts/feature_engineering.py
"""

from sqlalchemy import create_engine
from scripts.preprocessing.load_data import load_data
from scripts.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

# Establish database connection
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def feature_engineering(df):
    """
    Applies feature engineering to logistics data.

    Parameters:
    - df (pandas.DataFrame): Raw data from MariaDB.

    Returns:
    - pandas.DataFrame: Processed dataset.
    """
    # Columns to normalize using Min-Max Scaling
    numeric_cols = [
        "vehicle_gps_latitude", "vehicle_gps_longitude", "fuel_consumption_rate",
        "eta_variation_hours", "traffic_congestion_level", "warehouse_inventory_level",
        "loading_unloading_time", "handling_equipment_efficiency"
    ]

    # Check for missing columns
    existing_cols = [col for col in numeric_cols if col in df.columns]
    missing_cols = [col for col in numeric_cols if col not in df.columns]

    if missing_cols:
        print(f"[AVISO] As seguintes colunas estão ausentes e serão ignoradas: {missing_cols}")

    # Normalize only existing columns
    for col in existing_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Create new feature: traffic_density, if required columns exist
    if "traffic_congestion_level" in df.columns and "warehouse_inventory_level" in df.columns:
        df["traffic_density"] = df["traffic_congestion_level"] * df["warehouse_inventory_level"]
    else:
        print("[AVISO] Não foi possível criar 'traffic_density' por falta de colunas necessárias.")

    return df

def save_processed_data(df):
    """
    Saves processed data into MariaDB.

    Parameters:
    - df (pandas.DataFrame): Processed dataset.
    """
    try:
        df.to_sql("logistics_data_processed", engine, if_exists="replace", index=False)
        print("Processed data successfully saved to MariaDB (logistics_data_processed).")
    except Exception as e:
        print(f"Error saving processed data: {e}")

if __name__ == "__main__":
    df_raw = load_data()

    if df_raw is not None:
        print(f"Data successfully loaded! Total records: {df_raw.shape[0]}")

        df_processed = feature_engineering(df_raw)
        save_processed_data(df_processed)

        print("Feature engineering complete.")
        print(df_processed.head())


# """
# VI Manjaro-feature_engineering.py
# ------------------------
# Author: Jennifer Farias Rodrigues
# Date: 05/03/2025
# Description: This script performs feature engineering on logistics data and saves the processed data back to MariaDB.
#
# Structure:
# - Load raw data from MariaDB using `load_data.py`
# - Normalize numeric features using Min-Max Scaling
# - Create new feature: Traffic Density
# - Save the processed data back to MariaDB
#
# Usage:
# python scripts/feature_engineering.py
# """
#
# import pandas as pd
# from sqlalchemy import create_engine
# from scripts.load_data import load_data
# from scripts.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
#
# # Establish database connection
# engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
#
# def feature_engineering(df):
#     """
#     Applies feature engineering to logistics data.
#
#     Parameters:
#     - df (pandas.DataFrame): Raw data from MariaDB.
#
#     Returns:
#     - pandas.DataFrame: Processed dataset.
#     """
#     # Normalize numeric features using Min-Max Scaling
#     numeric_cols = [
#         "vehicle_gps_latitude", "vehicle_gps_longitude", "fuel_consumption_rate",
#         "eta_variation_hours", "traffic_congestion_level", "warehouse_inventory_level",
#         "loading_unloading_time", "handling_equipment_efficiency"
#     ]
#
#     for col in numeric_cols:
#         df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
#
#     # Create a new feature: Traffic Density
#     df["traffic_density"] = df["traffic_congestion_level"] * df["warehouse_inventory_level"]
#
#     return df
#
# def save_processed_data(df):
#     """
#     Saves processed data into MariaDB.
#
#     Parameters:
#     - df (pandas.DataFrame): Processed dataset.
#     """
#     try:
#         # Save the processed DataFrame to MariaDB
#         df.to_sql("logistics_data_processed", engine, if_exists="replace", index=False)
#         print("Processed data successfully saved to MariaDB (logistics_data_processed).")
#
#     except Exception as e:
#         print(f"Error saving processed data: {e}")
#
# if __name__ == "__main__":
#     # Load raw data
#     df_raw = load_data()
#
#     if df_raw is not None:
#         print(f"Data successfully loaded! Total records: {df_raw.shape[0]}")
#
#         # Apply feature engineering
#         df_processed = feature_engineering(df_raw)
#
#         # Save the processed data back to MariaDB
#         save_processed_data(df_processed)
#
#         print("Feature engineering complete.")
#         print(df_processed.head())
