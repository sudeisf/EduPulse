from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, trim, lower


def _normalize_score_columns(df: DataFrame, score_col: str) -> DataFrame:
    if score_col not in df.columns:
        return df

    return df.withColumn(
        score_col,
        when(col(score_col) < 0, 0.0)
        .when(col(score_col) > 100, 100.0)
        .otherwise(col(score_col).cast("double")),
    )


def clean_student_info(df: DataFrame) -> DataFrame:
    out = df

    if "final_result" in out.columns:
        out = out.withColumn("final_result", trim(col("final_result")))
    if "region" in out.columns:
        out = out.withColumn("region", trim(col("region")))
    if "highest_education" in out.columns:
        out = out.withColumn("highest_education", trim(col("highest_education")))

    if "gender" in out.columns:
        out = out.withColumn("gender", lower(trim(col("gender"))))

    if "studied_credits" in out.columns:
        out = out.withColumn(
            "studied_credits",
            when(col("studied_credits").isNull(), 0)
            .when(col("studied_credits") < 0, 0)
            .otherwise(col("studied_credits")),
        )

    return out


def clean_student_assessment(df: DataFrame) -> DataFrame:
    out = _normalize_score_columns(df, "score")

    if "date_submitted" in out.columns:
        out = out.withColumn(
            "date_submitted",
            when(col("date_submitted").isNull(), -1).otherwise(col("date_submitted")),
        )

    return out


def clean_assessments(df: DataFrame) -> DataFrame:
    out = _normalize_score_columns(df, "weight")
    return out


def clean_student_vle(df: DataFrame) -> DataFrame:
    out = df
    if "sum_click" in out.columns:
        out = out.withColumn(
            "sum_click",
            when(col("sum_click").isNull(), 0)
            .when(col("sum_click") < 0, 0)
            .otherwise(col("sum_click")),
        )
    return out


def apply_cleaning(dataset_name: str, df: DataFrame) -> DataFrame:
    if dataset_name == "studentInfo":
        return clean_student_info(df)
    if dataset_name == "studentAssessment":
        return clean_student_assessment(df)
    if dataset_name == "assessments":
        return clean_assessments(df)
    if dataset_name == "studentVle":
        return clean_student_vle(df)
    return df
