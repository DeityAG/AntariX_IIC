import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit app configuration
st.set_page_config(
    page_title="Codeplay-Satellite Orbit Predictor",
    page_icon="🛰️",
    layout="wide"
)

# Display the app title and logo
st.title("Codeplay-Satellite Orbit Predictor")
st.write("### 🛰️ Predicting Satellite Orbits with ML")

# Introduction text
st.write(
    """
    Welcome to the Codeplay Satellite Orbit Predictor. This tool allows you to input TLE (Two-Line Element) data 
    to predict satellite orbits with precision. Use the sections in the sidebar to navigate.
    """
)

# Sidebar for additional functionalities
st.sidebar.header("Navigation")
st.sidebar.write("Navigate to different sections:")
nav_options = ["Input TLE Data", "About the Model", "Contact"]
selected_option = st.sidebar.radio("Choose an option:", nav_options)

# Handle navigation
if selected_option == "Input TLE Data":

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
            st.session_state['tle_data'] = tle_input.strip()
            
            # Placeholder for predictions (to be replaced with your ML model)
            predictions = {
                "Satellite": ["ISS (ZARYA)", "CSS (TIANHE)", "ISS (NAUKA)", "FREGAT DEB"],
                "Predicted Altitude (km)": [420, 400, 430, 370],  # Dummy data
                "Predicted Velocity (km/s)": [7.66, 7.7, 7.68, 7.5],  # Dummy data
                "Orbital Inclination (°)": [51.6, 41.4, 51.6, 51.6],  # Dummy data
            }
            df = pd.DataFrame(predictions)
            
            # Display results
            st.header("Prediction Results")
            st.dataframe(df)
            
            # Visualize results
            st.write("### Graph of Predictions")
            fig = px.bar(
                df,
                x="Satellite",
                y="Predicted Altitude (km)",
                title="Altitude Prediction per Satellite",
                labels={"Predicted Altitude (km)": "Altitude (km)"}
            )
            st.plotly_chart(fig)
        else:
            st.error("Please input or upload TLE data before running predictions.")

elif selected_option == "About the Model":
    st.header("About the Model")
    st.write(
        """
        This tool uses a combination of the Simplified General Perturbations (SGP4) model and Machine Learning 
        to predict satellite orbits. You can paste TLE data to visualize predictions.
        """
    )

elif selected_option == "Contact":
    st.header("Contact")
    st.write(
        """
        For support or collaboration, please reach out to the development team.
        """
    )
