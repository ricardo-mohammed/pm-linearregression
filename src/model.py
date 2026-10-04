# src/model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class RegressionAnalyzer:
    def __init__(self, axis_columns):
        """
        Initialize the analyzer.
        :param axis_columns: List of the 8 axis column names.
        """
        self.axis_cols = axis_columns
        self.models = {}
        self.slopes = {}
        self.intercepts = {}
        self.residual_stds = {}
        
        # Thresholds (To be discovered)
        self.min_c = {}
        self.max_c = {}
        self.T = 5 # Time window in seconds (Assuming 1 row = 1 second based on synthetic data)

    def fit_models(self, df_train):
        """Fits univariate linear regression for Time -> Each Axis."""
        print("🔄 Fitting 8 Linear Regression models...")
        X_train = df_train['Time'].values.reshape(-1, 1)
        
        for col in self.axis_cols:
            y_train = df_train[col].values
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            self.models[col] = model
            self.slopes[col] = model.coef_[0]
            self.intercepts[col] = model.intercept_
            
        print(f"✅ Successfully fitted models for {len(self.models)} axes.")

    def analyze_residuals_and_discover_thresholds(self, df_train):
        """
        Calculates residuals, plots them, and uses Z-scores (Standard Deviation)
        to discover MinC and MaxC thresholds.
        """
        print("🔄 Analyzing residuals and discovering thresholds...")
        X_train = df_train['Time'].values.reshape(-1, 1)
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(self.axis_cols):
            y_train = df_train[col].values
            y_pred = self.models[col].predict(X_train)
            
            # Calculate Residuals (Actual - Predicted)
            residuals = y_train - y_pred
            
            # We only care about positive deviations (drawing MORE current than expected)
            # Calculate the standard deviation of the POSITIVE residuals to set our Z-score thresholds
            positive_residuals = residuals[residuals > 0]
            std_dev = np.std(positive_residuals) if len(positive_residuals) > 0 else np.std(residuals)
            self.residual_stds[col] = std_dev
            
            # Discover Thresholds using Z-scores
            # MinC (Alert) = 2 Standard Deviations above the line
            # MaxC (Error) = 3 Standard Deviations above the line
            self.min_c[col] = 2 * std_dev
            self.max_c[col] = 3 * std_dev
            
            # Plotting the residuals for the README/Report
            axes[i].hist(residuals, bins=30, color='skyblue', edgecolor='black')
            axes[i].set_title(f"{col} Residuals\nMinC: {self.min_c[col]:.2f} | MaxC: {self.max_c[col]:.2f}")
            axes[i].axvline(self.min_c[col], color='orange', linestyle='--', label='Alert (2σ)')
            axes[i].axvline(self.max_c[col], color='red', linestyle='--', label='Error (3σ)')
            axes[i].axvline(0, color='black', linestyle='-')
            axes[i].legend(fontsize=8)
            
        plt.tight_layout()
        plt.show()
        print("✅ Thresholds discovered and residual distributions plotted.")

    def evaluate_stream(self, df_stream):
        """
        Evaluates the streaming data, calculates real-time residuals,
        and applies the continuous time-window (T) logic to trigger Alerts/Errors.
        """
        print(f"🔄 Evaluating streaming data with T={self.T}s window...")
        
        # Dictionary to keep track of continuous violation times for each axis
        # Format: { 'Axis #1': {'alert_time': 0, 'error_time': 0, 'last_violation_time': -99} }
        violation_trackers = {
            col: {'alert_time': 0, 'error_time': 0, 'last_violation_time': -99} 
            for col in self.axis_cols
        }
        
        events_log = [] # To store triggered alerts/errors
        
        X_stream = df_stream['Time'].values.reshape(-1, 1)
        
        for idx, row in df_stream.iterrows():
            current_time = row['Time']
            
            for col in self.axis_cols:
                actual_val = row[col]
                predicted_val = self.models[col].predict([[current_time]])[0]
                residual = actual_val - predicted_val
                
                tracker = violation_trackers[col]
                
                # Check if the residual is above the regression line (positive)
                # and exceeds our thresholds
                if residual >= self.max_c[col]:
                    # ERROR Logic
                    if tracker['last_violation_time'] == current_time - 1:
                        tracker['error_time'] += 1
                    else:
                        tracker['error_time'] = 1
                        
                    tracker['last_violation_time'] = current_time
                    
                    if tracker['error_time'] >= self.T:
                        events_log.append({
                            'Time': current_time, 'Axis': col, 
                            'Type': 'ERROR', 'Residual': residual,
                            'Duration': tracker['error_time']
                        })
                        tracker['error_time'] = 0 # Reset after triggering
                        
                elif residual >= self.min_c[col]:
                    # ALERT Logic
                    if tracker['last_violation_time'] == current_time - 1:
                        tracker['alert_time'] += 1
                    else:
                        tracker['alert_time'] = 1
                        
                    tracker['last_violation_time'] = current_time
                    
                    if tracker['alert_time'] >= self.T:
                        events_log.append({
                            'Time': current_time, 'Axis': col, 
                            'Type': 'ALERT', 'Residual': residual,
                            'Duration': tracker['alert_time']
                        })
                        tracker['alert_time'] = 0 # Reset after triggering
                else:
                    # If it drops below the threshold, reset the continuous timers
                    tracker['alert_time'] = 0
                    tracker['error_time'] = 0

        print(f"✅ Stream evaluation complete. Detected {len(events_log)} events.")
        return pd.DataFrame(events_log)

    
    
    def plot_regression_with_alerts(self, df_stream, events_df):
        """
        Plots the regression lines with the synthetic data overlaid,
        and marks the exact locations of Alerts and Errors.
        """
        print("🔄 Generating final regression plots with Alert/Error overlays...")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        X_stream = df_stream['Time'].values.reshape(-1, 1)
        
        for i, col in enumerate(self.axis_cols):
            y_stream = df_stream[col].values
            y_pred = self.models[col].predict(X_stream)
            
            # Plot the synthetic data and the regression line
            axes[i].scatter(df_stream['Time'], y_stream, color='gray', alpha=0.3, s=10, label='Streamed Data')
            axes[i].plot(df_stream['Time'], y_pred, color='blue', linewidth=2, label='Regression Line')
            
            # Filter events for this specific axis
            axis_events = events_df[events_df['Axis'] == col]
            
            # Overlay ALERTS (Yellow triangles)
            alerts = axis_events[axis_events['Type'] == 'ALERT']
            if not alerts.empty:
                axes[i].scatter(alerts['Time'], y_stream[alerts.index], color='orange', marker='^', s=50, label=f'Alerts ({len(alerts)})', zorder=5)
                
            # Overlay ERRORS (Red stars)
            errors = axis_events[axis_events['Type'] == 'ERROR']
            if not errors.empty:
                axes[i].scatter(errors['Time'], y_stream[errors.index], color='red', marker='*', s=100, label=f'Errors ({len(errors)})', zorder=5)
            
            axes[i].set_title(f"{col}: Regression & Anomalies")
            axes[i].set_xlabel("Time (seconds)")
            axes[i].set_ylabel("Standardized Current")
            axes[i].legend(fontsize=8)
            
        plt.tight_layout()
        plt.show()
        print("✅ Final visualizations generated.")