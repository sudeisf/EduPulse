import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql.functions import col, sum as _sum, when, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from src.core.common import create_spark, read_processed_parquet, write_processed_parquet

def train_gpa_predictor():
    spark = create_spark("EduPulse-GPA-Regression", log_level="ERROR")

    # 1. Load Data
    # We need studentAssessment and assessments to calculate weighted scores
    student_assessments = read_processed_parquet(spark, "studentAssessment")
    assessments = read_processed_parquet(spark, "assessments")
    engagement = read_processed_parquet(spark, "engagement_features")

    # 2. Feature Engineering: Calculate Weighted Scores
    # Joining assessments to get weights for each test
    assessment_weights = student_assessments.join(assessments, "id_assessment")
    
    # Calculate weighted score: (score * weight) / 100
    weighted_df = assessment_weights.withColumn(
        "weighted_score", (col("score") * col("weight")) / 100
    )

    # Group by student to get their total current grade
    student_grades = weighted_df.groupBy("id_student").agg(
        _sum("weighted_score").alias("current_gpa")
    )

    # 3. Combine Grades with Engagement Metrics
    final_data = student_grades.join(engagement, "id_student")

    # Ensure required features are numeric and free of nulls for ML steps.
    final_data = final_data.select(
        "id_student",
        col("current_gpa").cast("double"),
        col("total_clicks").cast("double"),
        col("days_active").cast("double"),
        col("engagement_index").cast("double")
    )
    final_data = final_data.fillna({
        "total_clicks": 0.0,
        "days_active": 0.0,
        "engagement_index": 0.0
    }).dropna(subset=["current_gpa"])

    # 4. Prepare Machine Learning Features
    # Do not include the label (`current_gpa`) in features to avoid leakage.
    feature_cols = ["total_clicks", "days_active", "engagement_index"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    ml_ready_data = assembler.transform(final_data).select("features", col("current_gpa").alias("label"), "id_student")

    train_data, test_data = ml_ready_data.randomSplit([0.8, 0.2], seed=42)

    # 5. Train Linear Regression Model
    lr = LinearRegression(featuresCol="features", labelCol="label", regParam=0.1)
    lr_model = lr.fit(train_data)

    # 6. Generate Predictions
    # Evaluate on holdout data, but save predictions for all students so dashboard lookup works broadly.
    test_predictions = lr_model.transform(test_data)
    all_predictions = lr_model.transform(ml_ready_data)

    # Bound GPA to a valid percent range and keep metadata for UI explanations.
    bounded_predictions = all_predictions.withColumn(
        "predicted_gpa_raw",
        col("prediction")
    ).withColumn(
        "predicted_gpa",
        when(col("prediction") < 0, lit(0.0))
        .when(col("prediction") > 100, lit(100.0))
        .otherwise(col("prediction"))
    ).withColumn(
        "gpa_bounds_note",
        when(col("prediction") < 0, lit("capped_to_0"))
        .when(col("prediction") > 100, lit("capped_to_100"))
        .otherwise(lit("within_range"))
    )

    # Select relevant columns and save
    result_df = bounded_predictions.select("id_student", "predicted_gpa", "predicted_gpa_raw", "gpa_bounds_note")
    write_processed_parquet(result_df, "gpa_predictions")

    out_of_range_count = result_df.filter(col("gpa_bounds_note") != "within_range").count()
    print(f"Out-of-range GPA predictions capped: {out_of_range_count}")
    
    # Evaluate
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(test_predictions)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")

if __name__ == "__main__":
    train_gpa_predictor()