"""
IoTAnalisys.py
--------------
Author: Jennifer Farias Rodrigues
Date: 05/03/2025
Description: Main script for running data analysis on IoT logistics data.

Structure:
- Loads data from MariaDB using SQLAlchemy
- Performs exploratory data analysis (EDA)
- Prepares data for predictive modeling
- Calls machine learning models for demand forecasting

Usage:
python IoTAnalisys.py
"""

from scripts.preprocessing.load_data import load_data
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function to load and analyze logistics data.
    """
    print("Starting IoT Logistics Data Analysis...")

    # Load data from MariaDB
    df = load_data()

    if df is not None:
        print("\no Data Summary:")
        print(df.describe())  # Basic statistics
        print("\no First Rows of the Dataset:")
        print(df.head())

        # Save data to a local CSV (optional)
        df.to_csv("data/processed_logistics_data.csv", index=False)
        print("\no Data saved successfully in 'data/processed_logistics_data.csv'")

    else:
        print("\no Error: Data could not be loaded.")

if __name__ == "__main__":
    main()
