from common import create_spark, raw_path, write_processed_parquet
from cleaning import apply_cleaning

def ingest_raw_data():
    spark = create_spark(
        "EduPulse-Ingestion",
        extra_conf={"spark.sql.parquet.compression.codec": "snappy"},
    )

  
    files = ["studentVle.csv",
             "studentInfo.csv",
             "vle.csv",
             "studentAssessment.csv",
             "assessments.csv" ]

    for file in files:
        print(f"--- Processing {file} ---")
        # Load CSV
        df = spark.read.csv(raw_path(file), header=True, inferSchema=True)
        
        # Save as Parquet for high-performance reading later
        file_name = file.replace(".csv", "")
        cleaned_df = apply_cleaning(file_name, df)
        write_processed_parquet(cleaned_df, file_name)
        print(f"Saved {file_name} to Parquet format.")

if __name__ == "__main__":
    ingest_raw_data()