from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pyspark.sql import DataFrame, SparkSession


DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"


def create_spark(app_name: str, log_level: Optional[str] = None, extra_conf: Optional[Dict[str, str]] = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if extra_conf:
        for key, value in extra_conf.items():
            builder = builder.config(key, value)
    spark = builder.getOrCreate()
    if log_level:
        spark.sparkContext.setLogLevel(log_level)
    return spark


def raw_path(filename: str) -> str:
    return str(RAW_ROOT / filename)


def processed_path(filename: str) -> str:
    return str(PROCESSED_ROOT / filename)


def read_processed_parquet(spark: SparkSession, name: str):
    return spark.read.parquet(processed_path(f"{name}.parquet"))


def write_processed_parquet(df: DataFrame, name: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).parquet(processed_path(f"{name}.parquet"))


def normalize_id_column(df: pd.DataFrame, column: str = "id_student") -> pd.DataFrame:
    if column not in df.columns:
        return df

    numeric_ids = pd.to_numeric(df[column], errors="coerce")
    int_like_mask = numeric_ids.notna() & (numeric_ids % 1 == 0)
    df[column] = df[column].astype(str).str.strip()
    df.loc[int_like_mask, column] = numeric_ids[int_like_mask].astype("int64").astype(str)
    return df