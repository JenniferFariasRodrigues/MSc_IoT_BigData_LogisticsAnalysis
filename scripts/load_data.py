"""
load_data.py
--------------
Author: Jennifer Farias Rodrigues
Date: 05/03/2025
Description: This script connects to the MariaDB database, retrieves supply chain data,
             and prepares it for predictive analysis.

Structure:
- Establish connection to the MariaDB database
- Execute an SQL query to fetch the dataset
- Load the data into a Pandas DataFrame using SQLAlchemy
- Perform data cleaning and preprocessing
- Return the processed data for further analysis

Usage:
python scripts/load_data.py
"""

import pandas as pd
from sqlalchemy import create_engine  # Using SQLAlchemy for better database handling

# Database connection settings
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "jennifer"
DB_PASSWORD = ""
DB_NAME = "logistics_db"
TABLE_NAME = "logistics_data"

def load_data():
    """
    Connects to the MariaDB database using SQLAlchemy and retrieves logistics data.

    Returns:
    - pandas.DataFrame: Processed and ready-to-use data.
    """
    try:
        # Create SQLAlchemy engine
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

        print(f"Connected to MariaDB: {DB_NAME}")

        # Define SQL query to fetch data
        query = f"SELECT * FROM {TABLE_NAME};"

        # Load data into a Pandas DataFrame using SQLAlchemy
        df = pd.read_sql(query, engine)

        print(f"Data successfully loaded! Total records: {df.shape[0]}")

        # Drop missing values
        df.dropna(inplace=True)

        # Convert date columns to proper format (if applicable)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    except Exception as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

if __name__ == "__main__":
    # Execute only if the script is run directly
    df = load_data()
    if df is not None:
        print(df.head())  # Display first rows for verification
