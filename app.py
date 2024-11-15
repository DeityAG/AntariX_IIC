import streamlit as st
from st_pages import Page, show_pages, add_indentation, Section
from streamlit_lottie import st_lottie
import json

# Set page configuration
st.set_page_config(layout="wide", page_title="Orbit Predictor", page_icon="🌌")

# Set up sidebar with sections and pages
show_pages(
    [
        Section(name="Main Menu", icon="🏠"),
        Page("app.py", "Introduction", ""),
        Section(name="Modules", icon="🛰️"),
        Page("pages/orbit_prediction.py", "Orbit Prediction", ""),
        Page("pages/error_analysis.py", "Error Analysis", ""),
        Page("pages/visualization.py", "3D Visualization", ""),
        Section(name="Tools", icon="🛠️"),
        Page("pages/settings.py", "Settings", ""),
    ]
)
add_indentation()

# Sidebar info section
with st.sidebar:
    st.sidebar.title("About")
    st.sidebar.info(
        """
        🔗 [Your Project Website](https://your-project-link.com)

        ©️ 2024 Your Project Name
    """
    )

# Main content layout
st.title("Orbit Predictor - Hands on with Satellite Data")

# Load Lottie animation
with open('assets/lottie.json', 'r') as f:
    lottie_data = json.load(f)

# Layout with columns
col1, col2 = st.columns([0.3, 0.7])

with col1:
    st.write("""
        Welcome to the Orbit Predictor module. This tool helps you analyze satellite orbit data 
        using a hybrid model combining SGP4 with machine learning for improved accuracy.
        Explore our modules to start your satellite data journey!
    """)

with col2:
    st_lottie(lottie_data, key="orbit_animation", height=300, width=300)

# Hide Streamlit default menu and footer for a cleaner layout
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
