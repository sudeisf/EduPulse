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
    
@st.cache_data
def load_predictions():
    # Load the ML results
    pred_df = spark.read.parquet("data/processed/predictions.parquet")
    return pred_df.toPandas()

df_preds = load_predictions()

# Merge predictions with engagement for a complete view
df_final = pd.merge(df_engagement, df_preds, on="id_student")

# New Section: High Risk Alerts
st.subheader("⚠️ High-Risk Students (Priority Intervention)")
high_risk = df_final[df_final['risk_probability'] > 0.7].sort_values(by='risk_probability', ascending=False)

if not high_risk.empty:
    st.dataframe(high_risk[['id_student', 'engagement_index', 'risk_probability']].head(10))
else:
    st.success("No high-risk students detected at this threshold.")

# Update the Search Logic to show risk
if search_id:
    student_data = df_final[df_final['id_student'].astype(str) == search_id]
    if not student_data.empty:
        risk_val = student_data.iloc[0]['risk_probability']
        st.write(student_data)
        st.progress(float(risk_val))
        st.write(f"Predicted Risk Level: {risk_val*100:.1f}%")