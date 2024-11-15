import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit app configuration
st.set_page_config(
    page_title="AmtariX",
    page_icon="🛰️",
    layout="wide"
)

# Function to create a horizontal navigation bar
def nav_bar():
    st.markdown(
        """
        <style>
        .nav-bar {
            background-color: #f8f9fa;
            padding: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
        }
        .nav-bar a {
            text-decoration: none;
            color: #007bff;
            font-size: 18px;
            font-weight: bold;
        }
        .nav-bar a:hover {
            text-decoration: underline;
            color: #0056b3;
        }
        </style>
        <div class="nav-bar">
            <a href="/?nav=app" target="_self">App</a>
            <a href="/?nav=globe" target="_self">Globe Visualization</a>
            <a href="/?nav=research" target="_self">Research Work</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Determine the current page from query parameters
query_params = st.experimental_get_query_params()
current_page = query_params.get("nav", ["app"])[0]

# Display navigation bar
nav_bar()

# Page content
if current_page == "app":
    # App content
    st.image("assets/banner.jpg", use_column_width=True)  # Display the banner at the top
    st.title("AmtariX")
    st.write("### 🛰️ Predicting Satellite Orbits with ML")
    
    st.write(
        """
        Welcome to AmtariX. This tool allows you to input TLE (Two-Line Element) data 
        to predict satellite orbits with precision.
        """
    )
    
    # TLE Data Input Section
    st.header("Input TLE Data")
    st.write("Paste or upload your TLE data below:")
    
    # Text area for TLE input
    tle_input = st.text_area(
        "Enter TLE data (multiple satellites supported):",
        placeholder="Paste your TLE data here...",
        height=200
    )
    
    # File uploader for TLE data (optional)
    uploaded_file = st.file_uploader("Or upload a file with TLE data:", type=["txt"])
    
    if uploaded_file:
        tle_input = uploaded_file.read().decode("utf-8")
        st.success("TLE data uploaded successfully!")
    
    # Process the TLE input and generate predictions
    if st.button
