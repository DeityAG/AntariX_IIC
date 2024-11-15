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
st.sidebar.title("Navigation")
st.sidebar.write("Choose a page:")
page = st.sidebar.radio("", list(PAGES.keys()), index=0)

# Show selected page
if page == "Home":
    exec(open(PAGES[page]).read())
elif page

