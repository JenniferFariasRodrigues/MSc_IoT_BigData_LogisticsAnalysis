from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, window, max, to_timestamp

# Caminho do arquivo de dados (CSV)
file_path = "/home/jennifer/Documentos/Dissertação/spark/IoTLogisticsEnergyAnalysis/dynamic_supply_chain_logistics_dataset.csv"

# Criar a sessão Spark
spark = SparkSession.builder \
    .appName("IoT Logistics Energy Analysis") \
    .getOrCreate()

# Verificar se o arquivo existe antes de tentar carregá-lo
import os
if not os.path.exists(file_path):
    print(f"ERRO: O arquivo {file_path} não foi encontrado!")
    spark.stop()
    exit(1)

print("Arquivo encontrado, iniciando processamento...")

# Carregar o arquivo CSV no PySpark
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(file_path)

# Mostrar os primeiros registros do dataset
print("📌 Dados brutos carregados:")
df.show(5)

# Converter a coluna "timestamp" para formato de data-hora (se existir no CSV)
if "timestamp" in df.columns:
    df = df.withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))

# Remover
