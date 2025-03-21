import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME")
TABLE_NAME = os.getenv("TABLE_NAME")


def get_spark_session(app_name="SparkApp"):
    """
    Initialize and return a Spark session.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data_from_mariadb(spark):
    """
    Load processed data from MariaDB using JDBC connection.
    """
    jdbc_url = f"jdbc:mysql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
    properties = {
        "user": DB_USER,
        "password": DB_PASSWORD,
        "driver": "com.mysql.cj.jdbc.Driver"
    }
    return spark.read.jdbc(url=jdbc_url, table=TABLE_NAME, properties=properties)

