from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, sum as spark_sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pymongo import MongoClient

# Constants
MONGO_URI = 'mongodb://127.0.0.1:27017/'
MONGO_DB = 'sales_intelligence'
SPIKE_THRESHOLD = 50 # If sales in a 1-minute window exceed this, alert!

def write_streaming_alert_to_mongo(df, epoch_id):
    # This runs for every micro-batch
    records = df.collect()
    if records:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            db = client[MONGO_DB]
            
            alerts_to_insert = []
            for row in records:
                if row['total_sales'] > SPIKE_THRESHOLD:
                    alert = {
                        "product_id": row['product_id'],
                        "store_id": row['store_id'],
                        "alert_date": str(row['window']['end']),
                        "alert_type": "REALTIME_SPIKE",
                        "message": f"Real-time surge! {row['total_sales']} items sold in last minute.",
                        "severity": "CRITICAL"
                    }
                    alerts_to_insert.append(alert)
                    
            if alerts_to_insert:
                db.alerts.insert_many(alerts_to_insert)
                print(f"Inserted {len(alerts_to_insert)} real-time alerts.")
        except Exception as e:
            print(f"Warning: Could not write real-time alert to MongoDB: {e}")


def main():
    spark = SparkSession.builder \
        .appName("SalesRealtimeIntelligence") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # Define schema for incoming JSON data
    # Example format: {"timestamp": "2024-01-15T10:30:00", "product_id": "P1", "store_id": "S1", "sales": 5}
    schema = StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("product_id", StringType(), True),
        StructField("store_id", StringType(), True),
        StructField("sales", IntegerType(), True)
    ])

    print("Starting Streaming Context. Waiting for data on localhost:9999...")
    print("Hint: Use 'nc -lk 9999' to open socket and send JSON.")

    # Read from socket
    lines = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    # Parse JSON
    parsed = lines.select(from_json(col("value"), schema).alias("data")).select("data.*")

    # Aggregate sales over a 1-minute sliding window, updated every 30 seconds
    windowed_sales = parsed.groupBy(
        window(col("timestamp"), "1 minute", "30 seconds"),
        col("product_id"),
        col("store_id")
    ).agg(spark_sum("sales").alias("total_sales"))

    # Write stream out to console AND trigger MongoDB write for alerts
    query = windowed_sales.writeStream \
        .outputMode("complete") \
        .foreachBatch(write_streaming_alert_to_mongo) \
        .start()
        
    # Also print to console for debugging
    console_query = windowed_sales.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    query.awaitTermination()
    console_query.awaitTermination()

if __name__ == "__main__":
    main()
