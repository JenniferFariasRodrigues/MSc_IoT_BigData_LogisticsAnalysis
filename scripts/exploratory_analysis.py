"""
exploratory_analysis.py
------------------------
Author: Jennifer Farias Rodrigues
Date: 05/03/2025
Description: This script performs Exploratory Data Analysis (EDA) on logistics data using Pandas and PySpark.

Structure:
- Import the `load_data` function from load_data.py
- Convert the Pandas DataFrame to a PySpark DataFrame
- Display basic statistics (count, mean, std, min, max)
- Compute missing values per column
- Show sample records for verification
- Calculate correlation between numerical variables

Usage:
python scripts/exploratory_analysis.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
from pyspark.sql.functions import col, count
from scripts.load_data import load_data  # Importing the function instead of redefining it

# Initialize Spark session
spark = SparkSession.builder.appName("EDA_Logistics").getOrCreate()

# Load data using the function from load_data.py
df_pandas = load_data()

if df_pandas is not None:
    # Define schema explicitly for PySpark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("vehicle_gps_latitude", DoubleType(), True),
        StructField("vehicle_gps_longitude", DoubleType(), True),
        StructField("fuel_consumption_rate", DoubleType(), True),
        StructField("eta_variation_hours", DoubleType(), True),
        StructField("traffic_congestion_level", DoubleType(), True),
        StructField("warehouse_inventory_level", DoubleType(), True),
        StructField("loading_unloading_time", DoubleType(), True),
        StructField("handling_equipment_efficiency", DoubleType(), True),
    ])

    # Convert Pandas DataFrame to PySpark DataFrame using explicit schema
    df_spark = spark.createDataFrame(df_pandas, schema=schema)

    print(f"Data successfully loaded. Total records: {df_spark.count()}")

    # Display basic statistics
    df_spark.describe().show()

    # Count missing values per column
    df_spark.select([count(col(c)).alias(c) for c in df_spark.columns]).show()

    # Show sample records
    df_spark.show(5)

    # Compute correlations between numerical variables
    numeric_cols = [col_name for col_name, dtype in df_spark.dtypes if dtype in ('int', 'double')]
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                correlation = df_spark.stat.corr(col1, col2)
                print(f"Correlation between {col1} and {col2}: {correlation:.2f}")
