import streamlit as st

from dashboard_data import (
    load_analytics_data, 
    load_dashboard_data, 
    load_gpa_predictions
)
from dashboard_views import (
    render_executive_overview,
    render_institutional_analytics,
    render_student_search,
)


st.set_page_config(
    page_title="EduPulse Intelligence System",
    layout="wide",
)


df_main = load_dashboard_data()
df_gpa = load_gpa_predictions()
df_region, df_edu, df_region_gpa, df_edu_gpa = load_analytics_data()


st.sidebar.title("EduPulse Navigation")
page = st.sidebar.radio("Go to:", ["Executive Overview", "Student Search", "Institutional Analytics"])

if page == "Executive Overview":
    render_executive_overview(df_main, df_gpa)
elif page == "Student Search":
    render_student_search(df_main, df_gpa)
elif page == "Institutional Analytics":
    render_institutional_analytics(df_region, df_edu, df_region_gpa, df_edu_gpa)


st.sidebar.markdown("---")
st.sidebar.write("EduPulse v1.0 | Big Data Engine")