"""
cross_validation.py
------------------
Author: Jennifer Farias Rodrigues
Date: 20/03/2025
Description: Implements K-Fold cross-validation (k=2) to evaluate Random Forest and XGBoost models.
Uses Apache Spark for distributed processing of IoT data directly from MariaDB.

Dependencies:
- Requires `load_data()` from `load_data.py` to load the dataset.

Input: Data from MariaDB.
Output: MAE (Mean Absolute Error) after cross-validation and model saved in 'cross_validation_results'.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorAssembler
from scripts.load_data import load_data
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("CrossValidation").getOrCreate()

# Load processed data from MariaDB
data = load_data()

# Convert pandas DataFrame to Spark DataFrame if necessary
if isinstance(data, pd.DataFrame):
    print("Converting pandas DataFrame to Spark DataFrame...")
    data = spark.createDataFrame(data)

# Ensure `data` is a Spark DataFrame
if not isinstance(data, DataFrame):
    raise TypeError(f"Expected Spark DataFrame, but got {type(data)}")

# Check if data is empty
if data.count() == 0:
    print("Error: No data loaded. Check the database connection.")
    exit(1)

# Print column names for debugging
print("Columns found in the database:", data.columns)

# Define the correct target column based on database schema
TARGET_VARIABLE = "fuel_consumption_rate"

# Validate if target column exists
if TARGET_VARIABLE not in data.columns:
    raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset. Available columns: {data.columns}")

# Remove non-numeric columns if necessary
numeric_cols = [col for col, dtype in data.dtypes if dtype in ("double", "int", "float")]
if TARGET_VARIABLE not in numeric_cols:
    raise ValueError(f"Target variable '{TARGET_VARIABLE}' must be numeric. Found type: {dict(data.dtypes)[TARGET_VARIABLE]}")

# Ensure we have at least 2 rows for cross-validation
if data.count() < 5:
    print("⚠️ Warning: Not enough data for 5-fold cross-validation. Reducing to 2 folds.")
    numFolds = 2
else:
    numFolds = 5  # Default K-Fold value

# Create feature vector from available columns
feature_columns = [col for col in numeric_cols if col != TARGET_VARIABLE]

# Ensure there are features to train the model
if not feature_columns:
    raise ValueError("No valid numeric feature columns found for model training.")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", TARGET_VARIABLE)

# Define machine learning models
rf = RandomForestRegressor(featuresCol="features", labelCol=TARGET_VARIABLE, numTrees=100, maxDepth=6)
xgb = GBTRegressor(featuresCol="features", labelCol=TARGET_VARIABLE, maxIter=100, maxDepth=6)

# Set up cross-validation
paramGrid = ParamGridBuilder().build()  # Could include hyperparameter variations
evaluator = RegressionEvaluator(labelCol=TARGET_VARIABLE, metricName="mae")

crossval = CrossValidator(
    estimator=rf,  # Switch between rf and xgb to validate models
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=numFolds  # Adjusted K-Fold based on dataset size
)

# Apply cross-validation to the selected model
cvModel = crossval.fit(data)

# Evaluate the model using the MAE metric
predictions = cvModel.transform(data)
mae = evaluator.evaluate(predictions)

# Display the mean absolute error (MAE) result
print(f"Mean Absolute Error (MAE) after {numFolds}-Fold cross-validation: {mae}")

# Save the model trained with cross-validation
cvModel.write().overwrite().save("cross_validation_results")
