import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
import os

# Page Configuration
st.set_page_config(
    page_title="EduPulse Intelligence System",
    layout="wide"
)

# Initialize Spark Session
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("EduPulse-Dashboard") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

spark = get_spark()

# --- DATA LOADING FUNCTIONS (With String Casting Fix) ---

@st.cache_data
def load_dashboard_data():
    # Load primary engagement data
    df = spark.read.parquet("data/processed/engagement_features.parquet").toPandas()
    df['id_student'] = df['id_student'].astype(str) # Force to string
    
    # Merge Risk Predictions (Classification)
    path_risk = "data/processed/predictions.parquet"
    if os.path.exists(path_risk):
        risk_df = spark.read.parquet(path_risk).toPandas()
        risk_df['id_student'] = risk_df['id_student'].astype(str) # Force to string
        df = pd.merge(df, risk_df, on="id_student", how="left")
    else:
        df['risk_probability'] = 0.0
        df['prediction'] = 0
    return df

@st.cache_data
def load_gpa_predictions():
    # Load GPA Predictions (Regression)
    path_gpa = "data/processed/gpa_predictions.parquet"
    if os.path.exists(path_gpa):
        df = spark.read.parquet(path_gpa).toPandas()
        df['id_student'] = df['id_student'].astype(str) # Force to string
        return df
    return None

@st.cache_data
def load_analytics_data():
    region_stats = None
    edu_stats = None
    if os.path.exists("data/processed/region_stats.parquet"):
        region_stats = spark.read.parquet("data/processed/region_stats.parquet").toPandas()
    if os.path.exists("data/processed/education_stats.parquet"):
        edu_stats = spark.read.parquet("data/processed/education_stats.parquet").toPandas()
    return region_stats, edu_stats

# --- APP INITIALIZATION ---

df_main = load_dashboard_data()
df_gpa = load_gpa_predictions()
df_region, df_edu = load_analytics_data()

# Sidebar Navigation
st.sidebar.title("EduPulse Navigation")
page = st.sidebar.radio("Go to:", ["Executive Overview", "Student Search", "Institutional Analytics"])

# --- PAGE 1: EXECUTIVE OVERVIEW ---
if page == "Executive Overview":
    st.title("Executive Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df_main))
    with col2:
        avg_risk = df_main['risk_probability'].mean() * 100
        st.metric("Average Risk Level", f"{avg_risk:.1f}%")
    with col3:
        avg_eng = df_main['engagement_index'].mean()
        st.metric("Avg Engagement", round(avg_eng, 2))
    with col4:
        high_risk_count = len(df_main[df_main['risk_probability'] > 0.7])
        st.metric("High-Risk Alerts", high_risk_count)

    st.markdown("---")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Student Engagement Distribution")
        fig = px.histogram(df_main, x="engagement_index", nbins=30, color_discrete_sequence=['#4A90E2'])
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        st.subheader("Top High-Risk Students")
        top_risk = df_main.sort_values(by="risk_probability", ascending=False).head(10)
        st.dataframe(top_risk[['id_student', 'engagement_index', 'risk_probability']])

# --- PAGE 2: STUDENT SEARCH ---
elif page == "Student Search":
    st.title("Student Intelligence Lookup")
    st.markdown("Enter a student ID to retrieve cross-model analysis (Classification and Regression).")

    search_id = st.text_input("Enter Student ID (e.g., 11391):").strip()

    if search_id:
        # Match student in main dataframe
        student_row = df_main[df_main['id_student'] == search_id]
        
        if not student_row.empty:
            res = student_row.iloc[0]
            
            # GPA Prediction Lookup Logic
            current_gpa_val = "N/A"
            numeric_gpa = None
            if df_gpa is not None:
                gpa_match = df_gpa[df_gpa['id_student'] == search_id]
                if not gpa_match.empty:
                    numeric_gpa = float(gpa_match.iloc[0]['predicted_gpa'])
                    current_gpa_val = f"{numeric_gpa:.2f}%"

            st.markdown(f"### Results for Student ID: {search_id}")
            
            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                risk_p = float(res.get('risk_probability', 0))
                st.metric("Risk Probability", f"{risk_p*100:.1f}%")
                st.progress(risk_p)
            with m2:
                st.metric("Predicted GPA", current_gpa_val)
            with m3:
                st.metric("Engagement Index", round(res['engagement_index'], 2))
            with m4:
                st.metric("Activity Level", int(res['total_clicks']), "Clicks")

            # Status Messages based on Regression result
            if numeric_gpa is not None:
                if numeric_gpa < 50:
                    st.error(f"Critical: Predicted Grade ({current_gpa_val}) is below passing threshold.")
                elif numeric_gpa < 70:
                    st.warning(f"Borderline: Predicted Grade ({current_gpa_val}) requires academic support.")
                else:
                    st.success(f"Excellent: Predicted Grade ({current_gpa_val}) indicates high performance.")

            st.markdown("---")
            
            # Comparison Chart
            st.subheader("Student Behavior vs. Class Average")
            comp_df = pd.DataFrame({
                'Metric': ['Engagement', 'Total Clicks', 'Days Active'],
                'Student': [res['engagement_index'], res['total_clicks'], res['days_active']],
                'Average': [df_main['engagement_index'].mean(), df_main['total_clicks'].mean(), df_main['days_active'].mean()]
            })
            fig_comp = px.bar(comp_df, x='Metric', y=['Student', 'Average'], barmode='group',
                             color_discrete_map={'Student': '#4A90E2', 'Average': '#D3D3D3'})
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.error("Student ID not found. Verify that all Spark pipelines have been executed.")

# --- PAGE 3: INSTITUTIONAL ANALYTICS ---
elif page == "Institutional Analytics":
    st.title("Institutional Performance Analytics")
    
    if df_region is not None and df_edu is not None:
        t1, t2 = st.tabs(["Regional Analysis", "Demographic Analysis"])
        
        with t1:
            st.subheader("Regional Risk vs. Engagement")
            fig_reg = px.scatter(df_region, x="avg_engagement", y="avg_risk",
                                size="student_count", color="region", hover_name="region")
            st.plotly_chart(fig_reg, use_container_width=True)
            
        with t2:
            st.subheader("Risk Probability by Education Level")
            fig_edu = px.bar(df_edu, x="highest_education", y="avg_risk", color="avg_engagement")
            st.plotly_chart(fig_edu, use_container_width=True)
    else:
        st.warning("Analytics data missing. Run analytics.py to generate regional insights.")

st.sidebar.markdown("---")
st.sidebar.write("EduPulse v1.0 | Big Data Engine")