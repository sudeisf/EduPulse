import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql.functions import avg, count, col
from src.core.common import create_spark, read_processed_parquet, write_processed_parquet

def generate_demographic_insights():
    spark = create_spark("EduPulse-Analytics")

    # 1. Load Data
    student_info = read_processed_parquet(spark, "studentInfo")
    predictions = read_processed_parquet(spark, "predictions")
    engagement = read_processed_parquet(spark, "engagement_features")
    gpa_predictions = read_processed_parquet(spark, "gpa_predictions")

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

    # 5. GPA Trends by Region and Education
    gpa_base = student_info.join(gpa_predictions, "id_student")

    region_gpa_stats = gpa_base.groupBy("region").agg(
        avg("predicted_gpa").alias("avg_predicted_gpa"),
        count("id_student").alias("student_count")
    )

    education_gpa_stats = gpa_base.groupBy("highest_education").agg(
        avg("predicted_gpa").alias("avg_predicted_gpa"),
        count("id_student").alias("student_count")
    )

    # 6. Save Aggregated Insights
    write_processed_parquet(region_stats, "region_stats")
    write_processed_parquet(education_stats, "education_stats")
    write_processed_parquet(region_gpa_stats, "region_gpa_stats")
    write_processed_parquet(education_gpa_stats, "education_gpa_stats")
    
    print("Demographic analytics successfully generated.")

if __name__ == "__main__":
    generate_demographic_insights()