import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_echarts import st_echarts
import os

# Set page configuration
st.set_page_config(page_title="Orbit Predictor", page_icon="🛰️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Orbit Predictor")
    st.markdown("---")
    
    st.header("Navigation")
    # Basic radio buttons for navigation without using st_pages or any external library
    selected_page = st.radio("Go to", ["Introduction", "Orbit Prediction", "Error Analysis", "3D Visualization", "Settings"])

    st.markdown("### About")
    st.info(
        """
        This app allows you to visualize and predict satellite orbits using TLE data.
        For more information, visit [Your Project Website](https://your-project-link.com).
        """
    )

# Main content based on selected page
if selected_page == "Introduction":
    st.title("Orbit Predictor - Satellite Data Visualization")
    st.write("Welcome to the Orbit Predictor! Explore different modules to analyze satellite orbit data.")

elif selected_page == "Orbit Prediction":
    st.title("Orbit Prediction")
    st.write("Use this module to predict satellite orbits with the SGP4 and hybrid ML models.")
    tle_input = st.text_area("Enter TLE data for prediction", placeholder="Two-Line Element (TLE) data")
    if st.button("Run Prediction"):
        # Here you would add your orbit prediction logic
        st.write("Performing orbit prediction...")  # Placeholder for actual prediction function

elif selected_page == "Error Analysis":
    st.title("Error Analysis")
    st.write("Visualize error trends over different prediction horizons.")

    # Sample TLE Data (replace with actual data as needed)
    sample_data = {
        "OBJECT_NAME": ["STARLINK-1008", "STARLINK-1009", "STARLINK-1010", "STARLINK-1011", "STARLINK-1012"],
        "OBJECT_ID": ["2019-074B", "2019-074C", "2019-074D", "2019-074E", "2019-074F"],
        "EPOCH": ["2024-11-14T14:50:59", "2024-11-15T06:00:00", "2024-11-13T21:30:40", "2024-11-15T02:07:42", "2024-11-15T01:27:54"],
        "MEAN_MOTION": [15.06417543, 15.22839408, 15.06407753, 15.06417043, 15.06414096],
        "ECCENTRICITY": [0.0001594, 0.0001241, 0.0001563, 0.0001683, 0.0001441],
        "INCLINATION": [53.0523, 53.0538, 53.0537, 53.055, 53.0525]
    }
    df = pd.DataFrame(sample_data)

    # Display TLE Data
    st.subheader("TLE Data")
    st.write("The following TLE data is used for satellite orbit visualization and prediction.")
    st.dataframe(df)

    # Visualization of MEAN_MOTION trends
    st.subheader("Visualization: Mean Motion of Satellites")
    fig = px.line(df, x="OBJECT_NAME", y="MEAN_MOTION", title="Mean Motion of Satellites", markers=True)
    st.plotly_chart(fig)

elif selected_page == "3D Visualization":
    st.title("3D Visualization")
    
    # ECharts for interactive visualization
    st.subheader("Interactive Visualization (ECharts)")

    # Custom EChart configuration for eccentricity vs inclination
    options = {
        "title": {"text": "Eccentricity vs Inclination"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": df["OBJECT_NAME"].tolist()},
        "yAxis": {"type": "value"},
        "series": [
            {
                "name": "Eccentricity",
                "type": "line",
                "data": df["ECCENTRICITY"].tolist(),
                "smooth": True,
                "lineStyle": {"color": "red"},
            },
            {
                "name": "Inclination",
                "type": "line",
                "data": df["INCLINATION"].tolist(),
                "smooth": True,
                "lineStyle": {"color": "blue"},
            },
        ],
    }

    # Display EChart using streamlit-echarts library
    try:
        st_echarts(options=options, height="400px")
    except Exception as e:
        st.error(f"Error displaying EChart: {e}")

elif selected_page == "Settings":
    st.title("Settings")
    st.write("Configure application settings here.")

# Footer with credits
st.markdown("---")
st.markdown(
    '<h6>Developed by [Your Name](https://twitter.com/yourprofile) using Streamlit and ECharts</h6>',
    unsafe_allow_html=True,
)
