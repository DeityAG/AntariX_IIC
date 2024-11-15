import streamlit as st
import os

# Set title and load logo
st.title("Codeplay-Satellite Orbit Predictor")
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Introduction text
st.write("Welcome to the Codeplay Satellite Orbit Predictor. This tool allows you to input TLE data to predict satellite orbits with precision. Use the sections in the sidebar to navigate.")

# Input for TLE data
st.header("Input TLE Data")
tle_input = st.text_area("Enter TLE data here:", placeholder="Paste your TLE data...")

# Run button
if st.button("Run Prediction"):
    # Logic to save TLE data or pass it to backend (e.g., pickle model) would go here
    st.write("Processing TLE data...")
    # Simulate passing data to another page or backend
    st.session_state['tle_data'] = tle_input  # Store input in session state for use in Results page
    st.success("TLE data received. Go to the Results page to view predictions.")
