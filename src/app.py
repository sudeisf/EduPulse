import streamlit as st
from pyspark.sql import SparkSession

st.title("EduPulse: Big Data Intelligence")

spark = SparkSession.builder \
     .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

# Reduce Spark log noise in container output.
spark.sparkContext.setLogLevel("ERROR")
    
st.write("Spark Session Active!")
st.write(f"Spark Version: {spark.version}")

data = [("Student A", 85), ("Student B", 92), ("Student C", 78)]
df = spark.createDataFrame(data, ["Name", "Score"])
st.write(df.toPandas())