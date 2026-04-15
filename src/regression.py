from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def train_gpa_predictor():
    spark = SparkSession.builder.appName("EduPulse-GPA-Regression").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Load Data
    # We need studentAssessment and assessments to calculate weighted scores
    student_assessments = spark.read.parquet("data/processed/studentAssessment.parquet")
    assessments = spark.read.parquet("data/processed/assessments.parquet")
    engagement = spark.read.parquet("data/processed/engagement_features.parquet")

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
    predictions = lr_model.transform(test_data)
    
    # Select relevant columns and save
    result_df = predictions.select("id_student", col("prediction").alias("predicted_gpa"))
    result_df.write.mode("overwrite").parquet("data/processed/gpa_predictions.parquet")
    
    # Evaluate
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")

if __name__ == "__main__":
    train_gpa_predictor()