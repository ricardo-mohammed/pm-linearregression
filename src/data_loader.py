# data_loader.py

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


class DatabaseManager:
    def __init__(self, env_file='.env'):
        """Initialize the database connection using credentials from the .env file."""
        load_dotenv(env_file)
        self.database_url = os.getenv('NEON_DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("NEON_DATABASE_URL not found in .env file!")
            
        # 🛡️ BULLETPROOF FIX: 
        # SQLAlchemy 2.0 defaults to 'psycopg' (v3) if the URL starts with 'postgresql://'.
        # Since we have 'psycopg2-binary' installed, we force it to use that driver.
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            
        # Create the SQLAlchemy engine
        self.engine = create_engine(self.database_url)
        print("✅ Successfully connected to Neon Database.")


    def upload_training_data(self, df, table_name="training_telemetry"):
        """Uploads a pandas DataFrame to the Neon database."""
        # if_exists='replace' will drop the table if it exists and create a new one
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        print(f"✅ Uploaded {len(df)} rows to table '{table_name}'.")

    def fetch_training_data(self, table_name="training_telemetry"):
        """Queries the database and returns the data as a pandas DataFrame."""
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, self.engine)
        print(f"✅ Fetched {len(df)} rows from table '{table_name}' for training.")
        return df

    def insert_streaming_row(self, row_dict, table_name="streaming_telemetry"):
        """Inserts a single row (used later for the streaming simulation)."""
        df_row = pd.DataFrame([row_dict])
        # append mode adds to the existing table without deleting old data
        df_row.to_sql(table_name, self.engine, if_exists='append', index=False)
        
    def close_connection(self):
        """Disposes of the database engine."""
        self.engine.dispose()
        print("🔌 Database connection closed.")