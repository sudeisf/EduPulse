from pyspark.sql import SparkSession
import os

def initialize_spark():
    return SparkSession.builder \
        .appName("EduPulse-Ingestion") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

def ingest_raw_data():
    spark = initialize_spark()
    raw_path = "data/raw/"
    processed_path = "data/processed/"

  
    files = ["studentVle.csv",
             "studentInfo.csv",
             "vle.csv",
             "studentAssessment.csv",
             "assessments.csv" ]

    for file in files:
        print(f"--- Processing {file} ---")
        # Load CSV
        df = spark.read.csv(f"{raw_path}{file}", header=True, inferSchema=True)
        
        # Save as Parquet for high-performance reading later
        file_name = file.replace(".csv", "")
        df.write.mode("overwrite").parquet(f"{processed_path}{file_name}.parquet")
        print(f"Saved {file_name} to Parquet format.")

if __name__ == "__main__":
    ingest_raw_data()