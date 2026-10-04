# src/streaming_simulator.py
import pandas as pd
import time
from src.data_loader import DatabaseManager
from src.model import RegressionAnalyzer

class StreamingSimulator:
    def __init__(self, db_manager, analyzer, csv_path, delay=0.05):
        self.db = db_manager
        self.analyzer = analyzer
        self.csv_path = csv_path
        self.delay = delay

    def run_stream(self):
        print(f"🔄 Starting stream simulation from {self.csv_path}...")
        
        # Read the synthetic data
        df_stream = pd.read_csv(self.csv_path)
        
        # 🛡️ CRITICAL FIX: Create the table structure FIRST so 'append' works!
        # We write an empty dataframe (head(0)) to just create the columns, then we will append.
        df_stream.head(0).to_sql("streaming_telemetry", self.db.engine, if_exists='replace', index=False)
        print("✅ Prepared 'streaming_telemetry' table for streaming inserts.")
        
        streamed_data = []
        
        for index, row in df_stream.iterrows():
            row_dict = row.to_dict()
            
            # 1. Insert into Neon Database
            self.db.insert_streaming_row(row_dict, table_name="streaming_telemetry")
            streamed_data.append(row_dict)
            
            # Simulate real-time delay
            time.sleep(self.delay)
            
            if (index + 1) % 100 == 0:
                print(f"   ⏳ Streamed {index + 1}/{len(df_stream)} rows to Neon...")

        print(f"✅ Stream complete! {len(streamed_data)} rows inserted into Neon.")
        
        # 2. Convert the streamed data back to a DataFrame for evaluation
        df_streamed = pd.DataFrame(streamed_data)
        
        # 3. Evaluate the stream for Alerts and Errors
        events_df = self.analyzer.evaluate_stream(df_streamed)
        
        return df_streamed, events_df