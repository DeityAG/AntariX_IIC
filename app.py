import streamlit as st

# Set page configuration
st.set_page_config(page_title="Codeplay-Satellite Orbit Predictor", page_icon="🛰️", layout="wide")

# Sidebar navigation
st.sidebar.title("Codeplay Satellite Orbit Predictor")
st.sidebar.markdown("---")
st.sidebar.write("Navigate to different sections:")
st.sidebar.write("1. Welcome - Input TLE Data")
st.sidebar.write("2. Results - View Predictions")
st.sidebar.write("3. 3D Visualization - Interactive Plot")
st.sidebar.write("4. LLM Chat - Query the Model")
