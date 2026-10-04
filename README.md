# Practical Lab 1 - Predictive Maintenance: Streaming Data Anomaly Detection with Linear Regression

## 1. Project Summary
This project extends an industrial data streaming pipeline to implement regression-based anomaly detection for **Predictive Maintenance**. By analyzing historical telemetry data (current draw across 8 axes over time), the system trains univariate linear regression models to establish a baseline of expected machine behavior. 

Instead of relying on arbitrary, hardcoded thresholds, the pipeline analyzes regression residuals to dynamically discover statistically significant anomaly thresholds using Z-scores. Finally, a streaming simulator ingests synthetically generated testing data, evaluates it in real-time against the discovered thresholds, and triggers **Alerts** (early warning) and **Errors** (imminent failure) to simulate a real-world industrial monitoring environment.

---

## 2. Setup & Installation Instructions

### Prerequisites
- Python 3.8+
- A Neon.tech PostgreSQL database (credentials provided separately)

### Environment Setup
1. **Clone the repository** and navigate to the project root.
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment:**
   - *Windows:* `.venv\Scripts\activate`
   - *macOS/Linux:* `source .venv/bin/activate`
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Database Configuration (.env)
For security reasons, the `.env` file containing the Neon PostgreSQL connection string is **not** committed to this repository. 
* **Submission Note:** The `.env` file has been zipped and submitted separately via email, along with the database password. 
* **To run this project:** Unzip the provided `.env` file and place it in the root directory of this project before running the Jupyter Notebook. Ensure you use the connection string's password included in the email.

---

## 3. Methodology: Regression & Alert Rules

### 3.1 Data Preprocessing & Synthetic Testing Data
The raw training data (`RMBR4-2_export_test.csv`) is ingested into the Neon database and pulled back into the application to ensure strict adherence to the pipeline requirements. The data is standardized using Z-scores (mean=0, std=1) based *strictly* on the training metadata to prevent data leakage. 

Using the training metadata (mean and standard deviation), a synthetic testing dataset is generated via a normal distribution. Continuous anomalies are intentionally injected into this synthetic stream to simulate real-world predictive maintenance failures.

### 3.2 Regression Modeling & Residual Analysis
8 separate univariate linear regression models are fitted (`Time` -> `Axis #1` through `Axis #8`). To evaluate the models, we analyze the **residuals** (the difference between the observed current and the predicted current). 

### 3.3 Threshold Discovery (Critical Thinking)
Per the assignment requirements, thresholds were **not hardcoded** as fixed values. Instead, they were dynamically discovered from the training data's residual distribution using statistical principles:

* **MinC (Alert Threshold) = 2σ**: Set at 2 standard deviations above the regression line. A deviation of this magnitude is statistically unusual (~2.5% probability in a normal distribution). In a **Predictive Maintenance** context, this serves as an early warning sign of component wear, increased friction, or minor electrical degradation, prompting scheduled inspection before failure occurs.
* **MaxC (Error Threshold) = 3σ**: Set at 3 standard deviations above the regression line. A deviation of this size is highly anomalous (~0.1% probability). In an industrial setting, this indicates an imminent, catastrophic failure state (e.g., a seized motor or short circuit), requiring an immediate automated Error flag to prevent equipment damage.
* **Time Window (T = 5 seconds)**: Industrial sensors are prone to transient electrical noise or momentary spikes. By requiring the deviation to persist continuously for T=5 seconds, we filter out false positives and ensure that only *sustained* anomalous behavior triggers an event.

---

## 4. Results & Visualizations

### 4.1 Residual Analysis & Threshold Discovery
The following plots show the distribution of residuals for the 8 axes. The vertical lines indicate the dynamically discovered `MinC` (Orange) and `MaxC` (Red) thresholds based on the standard deviation of the positive residuals.

<img width="1789" height="790" alt="Residuals" src="https://github.com/user-attachments/assets/df0ae91f-0fad-4879-8b40-254952ac6fa9" />


### 4.2 Streaming Simulation & Anomaly Detection
The final dashboard overlays the streamed synthetic testing data and the trained linear regression baselines. 
* **Blue Line:** The expected baseline current (Regression prediction).
* **Gray Dots:** The actual streamed synthetic sensor data.
* **Orange Triangles:** Triggered **Alerts** (deviations >= 2σ sustained for T seconds).
* **Red Stars:** Triggered **Errors** (deviations >= 3σ sustained for T seconds).

This visual confirmation proves that the regression-based anomaly detection successfully identifies the injected continuous anomalies, validating our threshold discovery process.

<img width="1986" height="990" alt="Regression Anomalies" src="https://github.com/user-attachments/assets/b1fcdbf5-d94e-4dee-be87-eb5c1e6757aa" />


### 4.3 Event Logging
All triggered predictive maintenance events are captured and logged into a structured CSV file (`data/alert_error_log.csv`) and inserted into the Neon `streaming_telemetry` table for auditing.

<img width="1347" height="575" alt="image" src="https://github.com/user-attachments/assets/36f00e5c-fbf6-4aa3-8bdf-d516c252413d" />


---

## 5. Project Structure

```text
├── .env                  # Database credentials (Not committed, submitted via email)
├── .gitignore
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── data/
│   ├── RMBR4-2_export_test.csv       # Raw training data
│   ├── synthetic_testing_stream.csv  # Generated testing data
│   └── alert_error_log.csv           # Logged alerts and errors
├── notebooks/
│   └── pm_linear_regression.ipynb    # Main execution notebook
└── src/
    ├── data_loader.py           # Neon DB connection and OOP data ingestion
    ├── preprocessing.py         # Scaling, Z-score standardization, synthetic data generation
    ├── model.py                 # Regression fitting, residual analysis, threshold discovery
    └── streaming_simulator.py   # CSV -> DB time-based streaming simulation
```
