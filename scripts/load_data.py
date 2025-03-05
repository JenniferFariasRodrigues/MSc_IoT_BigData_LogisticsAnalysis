import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", 3306)  # Default to 3306 if not set
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME")
TABLE_NAME = os.getenv("TABLE_NAME")

def load_data():
    """
    Connects to the MariaDB database using SQLAlchemy and retrieves logistics data.

    Returns:
    - pandas.DataFrame: Processed and ready-to-use data.
    """
    try:
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

        print(f"Connected to MariaDB: {DB_NAME}")

        query = f"SELECT * FROM {TABLE_NAME};"
        df = pd.read_sql(query, engine)

        print(f"Data successfully loaded! Total records: {df.shape[0]}")

        df.dropna(inplace=True)

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
