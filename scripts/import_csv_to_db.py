import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env
load_dotenv()

# Dados de conexão do .env
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
table_name = os.getenv("TABLE_NAME")

# Caminho do CSV
csv_path = "C:/Users/Jennifer/Downloads/archive/dynamic_supply_chain_logistics_dataset.csv"

# Lê o CSV
df = pd.read_csv(csv_path)

# Cria engine de conexão
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Insere os dados no banco
df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

print("✅ Dados importados com sucesso!")

