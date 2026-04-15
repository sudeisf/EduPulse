import os

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

from src.core.common import normalize_id_column, processed_path


@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("EduPulse-Dashboard") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()


spark = get_spark()


@st.cache_data
def load_dashboard_data():
    df = spark.read.parquet(processed_path("engagement_features.parquet")).toPandas()
    df = normalize_id_column(df, "id_student")

    path_risk = processed_path("predictions.parquet")
    if os.path.exists(path_risk):
        risk_df = spark.read.parquet(path_risk).toPandas()
        risk_df = normalize_id_column(risk_df, "id_student")
        df = pd.merge(df, risk_df, on="id_student", how="left")
    else:
        df["risk_probability"] = 0.0
        df["prediction"] = 0
    return df


@st.cache_data
def load_gpa_predictions():
    path_gpa = processed_path("gpa_predictions.parquet")

    if not os.path.exists(path_gpa):
        return None

    try:
        df = spark.read.parquet(path_gpa).toPandas()
    except Exception:
        return None

    if "id_student" not in df.columns:
        return None

    df = normalize_id_column(df, "id_student")

    if "predicted_gpa" in df.columns:
        df["predicted_gpa"] = pd.to_numeric(df["predicted_gpa"], errors="coerce")
    if "predicted_gpa_raw" in df.columns:
        df["predicted_gpa_raw"] = pd.to_numeric(df["predicted_gpa_raw"], errors="coerce")
    return df


@st.cache_data
def load_analytics_data():
    region_stats = None
    edu_stats = None
    region_gpa_stats = None
    edu_gpa_stats = None

    region_path = processed_path("region_stats.parquet")
    edu_path = processed_path("education_stats.parquet")
    region_gpa_path = processed_path("region_gpa_stats.parquet")
    edu_gpa_path = processed_path("education_gpa_stats.parquet")

    if os.path.exists(region_path):
        region_stats = spark.read.parquet(region_path).toPandas()
    if os.path.exists(edu_path):
        edu_stats = spark.read.parquet(edu_path).toPandas()
    if os.path.exists(region_gpa_path):
        region_gpa_stats = spark.read.parquet(region_gpa_path).toPandas()
    if os.path.exists(edu_gpa_path):
        edu_gpa_stats = spark.read.parquet(edu_gpa_path).toPandas()

    return region_stats, edu_stats, region_gpa_stats, edu_gpa_stats


def build_good_history_view(main_df, gpa_df, max_risk, min_gpa, min_days_active):
    if gpa_df is None or gpa_df.empty:
        return pd.DataFrame()

    merged_df = pd.merge(main_df, gpa_df[["id_student", "predicted_gpa"]], on="id_student", how="left")
    merged_df["predicted_gpa"] = pd.to_numeric(merged_df["predicted_gpa"], errors="coerce")

    return merged_df[
        (merged_df["risk_probability"] <= max_risk)
        & (merged_df["predicted_gpa"] >= min_gpa)
        & (merged_df["days_active"] >= min_days_active)
    ].sort_values(by=["risk_probability", "predicted_gpa"], ascending=[True, False])


def normalize_search_id(value):
    value_str = str(value).strip()
    value_num = pd.to_numeric(pd.Series([value_str]), errors="coerce").iloc[0]
    if pd.notna(value_num) and float(value_num).is_integer():
        return str(int(value_num))
    return value_str
