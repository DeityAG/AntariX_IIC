import streamlit as st
import pandas as pd
import plotly.express as px
from pages.model_predictions import compute_original_trajectory, evaluate_model, plot_single_satellite_with_original

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
    if st.button("Run Prediction"):
        if tle_input.strip():
            st.write("Processing TLE data...")
            
            # Placeholder for actual TLE data parsing and processing
            # Replace with the actual TLE processing function as needed
            tle_data = pd.DataFrame([{
                "name": "ISS (ZARYA)",
                "line1": "1 25544U 98067A   24320.41934031  .00017230  00000+0  31097-3 0  9991",
                "line2": "2 25544  51.6416 286.1371 0007839 218.0643 253.6102 15.49814552482001"
            }])  # Replace with real TLE processing
            
            # Generate original trajectory
            original_positions = compute_original_trajectory(tle_data, "ISS (ZARYA)", days=30)
            
            # Evaluate model and get prediction results
            results = evaluate_model(tle_data, prediction_days=[10, 20, 30])
            
            # Display results
            st.header("Prediction Results")
            st.write("### Error Metrics")
            for key, result in results.items():
                st.write(f"{key}: Mean Error = {result['mean_error']:.2f} km, Max Error = {result['max_error']:.2f} km")
            
            # Plot the results
            st.write("### Prediction Plots")
            plot_single_satellite_with_original(results, "ISS (ZARYA)", original_positions)
            st.pyplot()  # Display the matplotlib plots
            
        else:
            st.error("Please input or upload TLE data before running predictions.")

elif current_page == "globe":
    # Globe Visualization content
    st.title("Globe Visualization")
    st.write("### 🗺️ Visualizing Satellite Orbits Globally")
    st.write(
        """
        This section is under development and will provide an interactive 3D globe to visualize satellite orbits.
        """
    )

elif current_page == "research":
    # Research Work content
    st.title("Research Work")
    st.write("### 📚 Detailed Research on Satellite Orbit Prediction")
    st.write(
        """
        Learn about the research and methodologies used for satellite orbit prediction, including the hybrid 
        modeling approach combining SGP4 and machine learning.
        """
    )
