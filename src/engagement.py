from pyspark.sql.functions import col, count, countDistinct, sum as _sum
from common import create_spark, read_processed_parquet, write_processed_parquet

def calculate_engagement():
    spark = create_spark("EduPulse-Engagement")
    
    # Load the optimized Parquet data
    vle_interactions = read_processed_parquet(spark, "studentVle")
    
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
    write_processed_parquet(engagement_df, "engagement_features")
    print("Engagement Indexing Complete. Preview:")
    engagement_df.show(5)

if __name__ == "__main__":
    calculate_engagement()