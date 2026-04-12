import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px

# Page Config
st.set_page_config(page_title="EduPulse Dashboard", layout="wide")

@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("EduPulse-UI").getOrCreate()

spark = get_spark()

st.title("🎓 EduPulse: Student Performance Intelligence")
st.markdown("---")

# 1. Load the Processed Data
@st.cache_data
def load_data():
    # Load the engagement features we created in the last step
    df = spark.read.parquet("data/processed/engagement_features.parquet")
    # Convert to Pandas for easier plotting (Spark -> Pandas is fine for aggregated data)
    return df.toPandas()

df_engagement = load_data()

# 2. Key Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Students Tracked", len(df_engagement))
with col2:
    st.metric("Avg Engagement Score", round(df_engagement['engagement_index'].mean(), 2))
with col3:
    st.metric("Max Interactions", int(df_engagement['total_clicks'].max()))

# 3. Visualizations
st.subheader("Engagement Distribution")
fig = px.histogram(df_engagement, x="engagement_index", 
                   nbins=50, title="How Students are Engaging",
                   color_discrete_sequence=['#00CC96'])
st.plotly_chart(fig, use_container_width=True)

# 4. Student Search Table
st.subheader("Student Engagement Search")
search_id = st.text_input("Enter Student ID to check engagement:")
if search_id:
    student_data = df_engagement[df_engagement['id_student'].astype(str) == search_id]
    if not student_data.empty:
        st.write(student_data)
    else:
        st.warning("Student ID not found.")
else:
    st.write(df_engagement.head(10))