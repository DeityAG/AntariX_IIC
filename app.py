import streamlit as st

# Set page configuration
st.set_page_config(page_title="Codeplay Satellite Orbit Predictor", page_icon="🛰️", layout="wide")

# Define page names and their respective files
PAGES = {
    "Home": "pages/Home.py",
    "Globe Visualization": "pages/Globe_Visualization.py",
    "Research Work": "pages/Research_Work.py"
}

# Display page selection options at the top
page = st.selectbox("Navigate to:", options=list(PAGES.keys()))

# Load and display the selected page
with open(PAGES[page]) as f:
    exec(f.read())
