# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    def __init__(self, axis_columns):
        """
        Initialize the preprocessor.
        :param axis_columns: List of the 8 axis column names from your CSV.
        """
        # TODO: UPDATE THESE TO MATCH CSV COLUMN NAMES
        self.axis_cols = axis_columns 
        
        # Initialize Scalers (Min-Max and Z-Score/Standard)
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()


    def generate_synthetic_data(self, df_train, n_samples=1000):
        """
        Generates synthetic testing data based on the training data's metadata (mean/std).
        Time values continue from where the training data ended.
        Intentionally injects a continuous anomaly to ensure alerts/errors trigger later.
        """
        print("🔄 Generating synthetic testing data from training metadata...")
        
        # 1. Extract metadata (mean and std) from the raw training data
        means = df_train[self.axis_cols].mean()
        stds = df_train[self.axis_cols].std()
        
        # 2. Generate normal synthetic data using numpy
        synthetic_data = pd.DataFrame()
        
        # Time continues from where training left off (1 row = ~1 second)
        train_end_time = df_train['Time'].max()
        synthetic_data['Time'] = np.arange(train_end_time + 1, train_end_time + 1 + n_samples)
        
        for col in self.axis_cols:
            synthetic_data[col] = np.random.normal(loc=means[col], scale=stds[col], size=n_samples)
            
        # 3. INJECT ANOMALIES
        anomaly_start = 400
        anomaly_duration = 50
        
        synthetic_data.loc[anomaly_start:anomaly_start+anomaly_duration, self.axis_cols[0]] += (stds[self.axis_cols[0]] * 5)
        synthetic_data.loc[anomaly_start:anomaly_start+anomaly_duration, self.axis_cols[2]] += (stds[self.axis_cols[2]] * 6)
        
        print(f"✅ Generated {n_samples} rows. Injected continuous anomaly from index {anomaly_start} to {anomaly_start+anomaly_duration}.")
        return synthetic_data

    def fit_and_transform(self, df_train, df_synthetic):
        """
        Fits the normalization and standardization scalers on the TRAINING data,
        then applies those exact same parameters to BOTH datasets.
        """
        print("🔄 Fitting scalers on training data and transforming both datasets...")
        
        # We only scale the axis columns, not the Time column
        train_features = df_train[self.axis_cols].copy()
        synth_features = df_synthetic[self.axis_cols].copy()
        
        # 1. Fit on Training Data ONLY
        self.minmax_scaler.fit(train_features)
        self.standard_scaler.fit(train_features)
        
        # 2. Transform Training Data
        df_train_scaled = df_train.copy()
        df_train_scaled[self.axis_cols] = self.minmax_scaler.transform(train_features)
        # Note: If you need both MinMax and Standard for different parts of the pipeline, 
        # you can store them in separate columns, but usually Standard (Z-score) is best for residual analysis.
        # Let's overwrite with Standard (Z-score) for the regression, as it makes residuals easier to interpret.
        df_train_scaled[self.axis_cols] = self.standard_scaler.transform(train_features)
        
        # 3. Transform Synthetic Data using the TRAINING parameters
        df_synth_scaled = df_synthetic.copy()
        df_synth_scaled[self.axis_cols] = self.standard_scaler.transform(synth_features)
        
        print("✅ Data successfully normalized and standardized with respect to training data.")
        return df_train_scaled, df_synth_scaled

    # Add this method to your DataPreprocessor class in src/preprocessing.py

    def convert_timestamps(self, df):
        """
        Converts ISO 8601 timestamp strings in the 'Time' column 
        to numeric seconds elapsed since the first timestamp.
        """
        print("🔄 Converting timestamp strings to numeric seconds...")
        
        # 1. Convert the string to actual datetime objects
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 2. Calculate seconds elapsed since the very first timestamp
        start_time = df['Time'].min()
        df['Time'] = (df['Time'] - start_time).dt.total_seconds()
        
        print(f"✅ Time column converted. Range: 0 to {df['Time'].max():.1f} seconds.")
        return df