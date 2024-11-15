import streamlit as st
import pandas as pd
import plotly.express as px
import os

from streamlit_navigation_bar import st_navbar

page = st_navbar(["Home", "Documentation", "Examples", "Community", "About"])
st.write(page)

# Set page configuration
st.set_page_config(page_title="Codeplay-Satellite Orbit Predictor", page_icon="🛰️", layout="wide")

# Display the app name and logo on the main page
st.title("Codeplay-Satellite Orbit Predictor")
logo_path = "assets/isrologo1.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Introduction text
st.write("Welcome to the Codeplay Satellite Orbit Predictor. This tool allows you to input TLE data to predict satellite orbits with precision. Use the sections in the sidebar to navigate.")

# Input for TLE data
st.header("Input TLE Data")
tle_input = st.text_area("Enter TLE data here:", placeholder="Paste your TLE data...")

# Run Prediction Button
if st.button("Run Prediction"):
    # Placeholder for ML model - loading from a pickle file or other source
    st.write("Processing TLE data...")
    st.session_state['tle_data'] = tle_input  # Store input in session state

    # Placeholder for predictions (you will replace this with model predictions later)
    predictions = {
        "Parameter": ["Predicted Altitude", "Predicted Velocity", "Orbital Inclination"],
        "Value": [550, 7.8, 53.0]  # Dummy values
    }
    df = pd.DataFrame(predictions)

    # Display results section
    st.header("Results")
    st.write("### Prediction Results")
    st.dataframe(df)

    # Display simple graph
    st.write("### Graph of Predictions")
    fig = px.bar(df, x="Parameter", y="Value", title="Prediction Results")
    st.plotly_chart(fig)

else:
    st.info("Enter TLE data above and click 'Run Prediction' to view results.")

