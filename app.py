import streamlit as st

# Set page configuration
st.set_page_config(page_title="Codeplay Satellite Orbit Predictor", page_icon="🛰️", layout="wide")

# Sidebar Navigation
st.sidebar.title("Codeplay Satellite Orbit Predictor")
st.sidebar.markdown("---")
st.sidebar.write("Navigate to different sections:")
st.sidebar.write("1. Home - ML Model and Predictions")
st.sidebar.write("2. Globe Visualization - Interactive Plot")
st.sidebar.write("3. Research Work - Project Details")

# Load selected page
selected_page = st.sidebar.radio("Choose a page", ["Home", "Globe Visualization", "Research Work"])

# Redirect to the appropriate page file based on selection
if selected_page == "Home":
    import pages.Home
elif selected_page == "Globe Visualization":
    import pages.Globe_Visualization
elif selected_page == "Research Work":
    import pages.Research_Work
