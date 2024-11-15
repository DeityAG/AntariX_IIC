import streamlit as st

# Set page configuration
st.set_page_config(page_title="Codeplay Satellite Orbit Predictor", page_icon="🛰️", layout="wide")

# Define navigation pages
PAGES = {
    "Home": "pages/Home.py",
    "Globe Visualization": "pages/Globe_Visualization.py",
    "Research Work": "pages/Research_Work.py"
}

# Define navigation buttons
st.title("Codeplay Satellite Orbit Predictor")
st.markdown("---")
page = st.selectbox("Choose a page:", options=list(PAGES.keys()))

# Load and display the selected page
if page == "Home":
    exec(open(PAGES[page]).read())
elif page == "Globe Visualization":
    exec(open(PAGES[page]).read())
elif page == "Research Work":
    exec(open(PAGES[page]).read())
