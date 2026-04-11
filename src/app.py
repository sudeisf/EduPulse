import streamlit as st
from pyspark.sql import SparkSession

st.title("EduPulse: Big Data Intelligence")

spark = SparkSession.builder \
    .appName("EduPulseLocal") \
    .getOrCreate()
    
st.write("Spark Session Active!")
st.write(f"Spark Version: {spark.version}")

data = [("Student A", 85), ("Student B", 92), ("Student C", 78)]
df = spark.createDataFrame(data, ["Name", "Score"])
st.write(df.toPandas())