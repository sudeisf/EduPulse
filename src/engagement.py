from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, sum as _sum

def calculate_engagement():
    spark = SparkSession.builder.appName("EduPulse-Engagement").getOrCreate()
    
    # Load the optimized Parquet data
    vle_interactions = spark.read.parquet("data/processed/studentVle.parquet")
    
    print("Calculating Behavioral Metrics...")
    
    # Feature Engineering: Aggregate by Student
    engagement_df = vle_interactions.groupBy("id_student").agg(
        _sum("sum_click").alias("total_clicks"),
        countDistinct("date").alias("days_active"),
        count("id_site").alias("total_interactions")
    )
    
    # Calculate Engagement Index (Simple weighted formula)
    # Score = (Total Clicks * 0.4) + (Days Active * 0.6)
    # We normalize these later, but for now, let's build the raw index
    engagement_df = engagement_df.withColumn(
        "engagement_index", 
        (col("total_clicks") * 0.4) + (col("days_active") * 0.6)
    )
    
    # Save the features
    engagement_df.write.mode("overwrite").parquet("data/processed/engagement_features.parquet")
    print("Engagement Indexing Complete. Preview:")
    engagement_df.show(5)

if __name__ == "__main__":
    calculate_engagement()