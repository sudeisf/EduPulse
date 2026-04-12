from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col

def generate_demographic_insights():
    spark = SparkSession.builder.appName("EduPulse-Analytics").getOrCreate()

    # 1. Load Data
    student_info = spark.read.parquet("data/processed/studentInfo.parquet")
    predictions = spark.read.parquet("data/processed/predictions.parquet")
    engagement = spark.read.parquet("data/processed/engagement_features.parquet")

    # 2. Multi-table Join
    # Combine demographics, engagement metrics, and risk predictions
    combined_data = student_info.join(predictions, "id_student") \
                                .join(engagement, "id_student")

    # 3. Aggregate Performance by Region
    region_stats = combined_data.groupBy("region").agg(
        avg("risk_probability").alias("avg_risk"),
        avg("engagement_index").alias("avg_engagement"),
        count("id_student").alias("student_count")
    )

    # 4. Aggregate Performance by Education Level
    education_stats = combined_data.groupBy("highest_education").agg(
        avg("risk_probability").alias("avg_risk"),
        avg("engagement_index").alias("avg_engagement")
    )

    # 5. Save Aggregated Insights
    region_stats.write.mode("overwrite").parquet("data/processed/region_stats.parquet")
    education_stats.write.mode("overwrite").parquet("data/processed/education_stats.parquet")
    
    print("Demographic analytics successfully generated.")

if __name__ == "__main__":
    generate_demographic_insights()