from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from common import create_spark, read_processed_parquet, write_processed_parquet


def train_at_risk_model():
    spark = create_spark("EduPulse-ML")
    
    engagement = read_processed_parquet(spark, "engagement_features")
    student_info = read_processed_parquet(spark, "studentInfo")
    
    #  2. Prepare Labels (Target Variable)
    # Define At-Risk (1) as Fail or Withdrawn, Safe (0) as Pass or Distinction
    data = engagement.join(student_info, "id_student")
    data = data.withColumn("label", 
        when((col("final_result") == "Fail") | (col("final_result") == "Withdrawn"), 1).otherwise(0)
    )
    
    # 3. Feature Assembly
    # Spark MLlib requires all features in a single "features" vector column
    feature_cols = ["total_clicks", "days_active", "engagement_index"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    final_data = assembler.transform(data).select("features", "label", "id_student")

    # 4. Split Data (80% Training, 20% Testing)
    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

    # 5. Train Random Forest Model
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    model = rf.fit(train_data)

    # 6. Evaluate Model
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions)
    print(f"Model Accuracy (ROC): {accuracy:.2f}")

    # 7. Save Predictions for Dashboard
    # We apply the model to the WHOLE dataset to get risk probabilities for everyone
    all_predictions = model.transform(final_data)
    
    # Extract probability of failing (the second value in the probability vector)
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vector
    from pyspark.sql.types import FloatType

    get_risk_prob = udf(lambda v: float(v[1]), FloatType())
    
    result_df = all_predictions.withColumn("risk_probability", get_risk_prob("probability")) \
                               .select("id_student", "prediction", "risk_probability")

    write_processed_parquet(result_df, "predictions")
    print("Predictions saved to data/processed/predictions.parquet")

if __name__ == "__main__":
    train_at_risk_model()