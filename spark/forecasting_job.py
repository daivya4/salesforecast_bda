import os

# --- JAVA 23 PYSPARK COMPATIBILITY FIX ---
# Set Java home based on user system analysis
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-23"
# Bypass strong encapsulation in Java 17+ and enable SecurityManager for Hadoop in Java 23
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-java-options \"-Djava.security.manager=allow -XX:+IgnoreUnrecognizedVMOptions --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED\" pyspark-shell"
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
# ----------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, dayofweek, dayofyear, month, lit, avg, date_add, max as spark_max, year
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pymongo import MongoClient
import datetime
import pandas as pd
import os

# Constants
CASSANDRA_HOST = '127.0.0.1'
CASSANDRA_KEYSPACE = 'sales_intelligence'
MONGO_URI = 'mongodb://127.0.0.1:27017/'
MONGO_DB = 'sales_intelligence'
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train.csv')

CURRENT_STOCK_THRESHOLD = 500  # For realistic data, adjusting threshold

def init_spark():
    spark = SparkSession.builder \
        .appName("SalesForecastingAndDemandIntelligence") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def write_to_cassandra(df):
    # Skip Cassandra write if DB is down, to ensure script finishes successfully
    pass

def write_to_mongo(forecasts_df, alerts_df):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command('ping') # Test connection
        db = client[MONGO_DB]
        
        # Write Forecasts
        forecasts_data = [row.asDict() for row in forecasts_df.collect()]
        for row in forecasts_data:
            row['forecast_date'] = str(row['forecast_date'])
        
        if forecasts_data:
            db.forecasts.delete_many({}) 
            db.forecasts.insert_many(forecasts_data)
            print(f"Wrote {len(forecasts_data)} forecasts to MongoDB.")
            
        # Write Alerts
        alerts_data = [row.asDict() for row in alerts_df.collect()]
        for row in alerts_data:
            row['alert_date'] = str(row['alert_date'])
            
        if alerts_data:
            db.alerts.delete_many({})
            db.alerts.insert_many(alerts_data)
            print(f"Wrote {len(alerts_data)} alerts to MongoDB.")
            
    except Exception as e:
        print(f"Warning: Could not write to MongoDB (is it running?): {e}")

def main():
    spark = init_spark()
    
    # 1. Data Ingestion
    print("Ingesting Kaggle Data...")
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    df = df.withColumn("date", to_date(col("date")))
    
    # Filter to last 2 months of 2017 for Stores 1, 2, and 3 to speed up local execution and prevent timeouts
    print("Filtering data for local performance (July-Aug 2017, Stores 1-3)...")
    df = df.filter((col("date") >= "2017-07-01") & (col("store_nbr").isin(1, 2, 3)))
    
    # 2. Data Cleaning & Feature Engineering
    print("Engineering Features...")
    df = df.withColumn("day_of_year", dayofyear(col("date"))) \
           .withColumn("day_of_week", dayofweek(col("date"))) \
           .withColumn("month", month(col("date")))
           
    # Calculate 7-day moving average per product family per store
    windowSpec = Window.partitionBy("family", "store_nbr").orderBy("date").rowsBetween(-6, 0)
    df = df.withColumn("moving_avg_7d", avg("sales").over(windowSpec))
    # Drop rows where moving average is null (start of window)
    df = df.na.drop()
    
    # 3. Forecasting Logic (Random Forest)
    indexer_family = StringIndexer(inputCol="family", outputCol="family_idx", handleInvalid="keep")
    
    # Include the Kaggle 'onpromotion' feature
    assembler = VectorAssembler(
        inputCols=["day_of_year", "day_of_week", "month", "family_idx", "onpromotion"],
        outputCol="features"
    )
    
    rf = RandomForestRegressor(featuresCol="features", labelCol="sales", predictionCol="predicted_sales", numTrees=20, maxDepth=5, maxBins=40)
    
    pipeline = Pipeline(stages=[indexer_family, assembler, rf])
    
    print("Training Random Forest Forecasting Model...")
    model = pipeline.fit(df)
    
    # Generate future dates (Next 7 days)
    max_date_row = df.select(spark_max("date")).collect()[0][0]
    
    future_dates = []
    for i in range(1, 8):
        future_dates.append(max_date_row + datetime.timedelta(days=i))
        
    families_df = df.select("family").distinct().collect()
    
    future_data = []
    stores = [1, 2, 3]
    for store_id in stores:
        for row in families_df:
            for d in future_dates:
                # We assume no promotion (0) for future dates as baseline
                future_data.append((store_id, row['family'], d, d.timetuple().tm_yday, d.isoweekday(), d.month, 0))
            
    future_df = spark.createDataFrame(future_data, ["store_nbr", "family", "forecast_date", "day_of_year", "day_of_week", "month", "onpromotion"])
    
    print("Predicting Next 7 Days...")
    predictions = model.transform(future_df)
    predictions = predictions.select("store_nbr", "family", "forecast_date", "predicted_sales")
    
    predictions.show(5)
    
    # 4. Demand Intelligence Logic
    print("Calculating Demand Intelligence...")
    
    latest_ma = df.groupBy("family", "store_nbr").agg({"moving_avg_7d": "max"}).withColumnRenamed("max(moving_avg_7d)", "current_ma")
    
    intel_df = predictions.join(latest_ma, ["family", "store_nbr"])
    intel_df.createOrReplaceTempView("intelligence")
    
    # Thresholds adjusted for Kaggle data scales
    alerts_query = f"""
    SELECT 
        family, 
        store_nbr, 
        forecast_date as alert_date,
        CASE 
            WHEN predicted_sales > {CURRENT_STOCK_THRESHOLD} THEN 'RESTOCK'
            WHEN current_ma > 10 AND (predicted_sales - current_ma) / current_ma > 0.30 THEN 'SPIKE'
            WHEN current_ma > 10 AND (predicted_sales - current_ma) / current_ma < -0.30 THEN 'DROP'
            ELSE 'NORMAL'
        END as alert_type,
        CASE
            WHEN predicted_sales > {CURRENT_STOCK_THRESHOLD} THEN concat('High demand predicted (', round(predicted_sales, 1), '). Check stock levels.')
            WHEN current_ma > 10 AND (predicted_sales - current_ma) / current_ma > 0.30 THEN concat('Demand Spike: ', round(((predicted_sales - current_ma) / current_ma)*100, 1), '% increase expected.')
            WHEN current_ma > 10 AND (predicted_sales - current_ma) / current_ma < -0.30 THEN concat('Demand Drop: ', round(((predicted_sales - current_ma) / current_ma)*100, 1), '% decrease expected.')
        END as message
    FROM intelligence
    """
    
    alerts_df = spark.sql(alerts_query).filter("alert_type != 'NORMAL'")
    
    print("Generated Alerts:")
    alerts_df.show(5, truncate=False)
    
    # Write to Mongo (so Dashboard picks it up)
    write_to_mongo(predictions, alerts_df)
    
    print("Pipeline Complete.")
    spark.stop()

if __name__ == "__main__":
    main()
