# Sales Forecasting and Demand Intelligence System

A full-stack Big Data pipeline using Apache Spark, MongoDB, Cassandra, and Streamlit.

## Architecture
1. **Apache Spark (PySpark)** processes daily sales batches and streaming events to generate forecasts and detect demand anomalies.
2. **Cassandra** stores historical time-series sales data.
3. **MongoDB** stores analytics outcomes (predictions, alerts).
4. **Streamlit** provides an interactive dashboard to monitor metrics and alerts.

## Requirements
```bash
pip install -r requirements.txt
```

## Running the Project

### 1. (Optional) Start Databases
Ensure MongoDB is running on `localhost:27017` and Cassandra is running on `localhost:9042`. 
*(Note: the scripts have fallbacks and will still run using local CSVs and mock data if databases are unavailable!)*

### 2. Run the Batch Forecasting Job
This script trains a Linear Regression model, generates a 7-day forecast, and computes alerts.
```bash
python spark/forecasting_job.py
# or using spark-submit
# spark-submit spark/forecasting_job.py
```

### 3. Start the Dashboard
Visualize the historical data, predictions, and alerts.
```bash
streamlit run dashboard/app.py
```

### 4. (Bonus) Real-time Streaming
To test real-time demand spike detection:
1. Open a new terminal and start a socket server:
   ```bash
   nc -lk 9999
   ```
2. Start the streaming job:
   ```bash
   python spark/streaming_job.py
   ```
3. Paste JSON records into the `nc` terminal:
   ```json
   {"timestamp": "2024-01-15T10:30:00", "product_id": "P1", "store_id": "S1", "sales": 60}
   ```
4. Observe the console output and check the dashboard for `REALTIME_SPIKE` alerts!
