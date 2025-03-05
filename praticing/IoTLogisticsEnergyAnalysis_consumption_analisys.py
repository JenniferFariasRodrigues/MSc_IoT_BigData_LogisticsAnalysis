
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, window, max, to_timestamp

# Initialize Spark Session
spark = SparkSession.builder.appName("IoT Logistics Energy Analysis").getOrCreate()

# Load the dataset from the specified path
file_path = "/home/jennifer/Documentos/Dissertação/spark/IoTLogisticsEnergyAnalysis/dynamic_supply_chain_logistics_dataset.xlsx"
df = spark.read.format("com.crealytics.spark.excel").option("header", "true").option("inferSchema", "true").load(file_path)

# Clean the data by removing null values and duplicates
df = df.na.drop().dropDuplicates()
# Convert timestamp column to the correct format
df = df.withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))

# Perform Peak Consumption Analysis by device and hour
df_grouped = df.groupBy("device_id", window("timestamp", "1 hour")).agg(max("energy_usage").alias("peak_consumption"))
df_grouped.show()

# Placeholder for Future Consumption Forecasting Implementation

# Perform Efficiency Analysis by calculating average consumption by device and location
df_avg = df.groupBy("device_id", "location").agg(avg("energy_usage").alias("avg_consumption"))
df_avg.show()

# Stop the Spark session
spark.stop()
