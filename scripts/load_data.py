"""
load_data.py
--------------
Author: Jennifer Farias Rodrigues
Date: 05/03/2025
Description: This script connects to the MariaDB database using SQLAlchemy,
retrieves logistics data, and prepares it for predictive analysis.

Structure:
- Import database connection settings from config.py
- Establish connection to the MariaDB database
- Execute an SQL query to fetch the dataset
- Load the data into a Pandas DataFrame
- Perform basic data cleaning
- Return the processed data for further analysis

Usage:
python scripts/load_data.py
"""

import pandas as pd
from sqlalchemy import create_engine
from scripts.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, TABLE_NAME

def load_data():
    """
    Connects to the MariaDB database using SQLAlchemy and retrieves logistics data.

    Returns:
    - pandas.DataFrame: Processed and ready-to-use data.
    """
    try:
        # Establish database connection
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        print(f"Connected to MariaDB: {DB_NAME}")

        # Execute query and load data into Pandas DataFrame
        query = f"SELECT * FROM {TABLE_NAME};"
        df = pd.read_sql(query, engine)

        print(f"Data successfully loaded! Total records: {df.shape[0]}")

        # Drop missing values
        df.dropna(inplace=True)

        # Convert date columns to datetime format if applicable
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    except Exception as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(df.head())
