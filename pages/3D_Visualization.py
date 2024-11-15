import streamlit as st

st.title("3D Visualization")
st.write("Interactive 3D Satellite Orbit Visualization with Cesium.js")

# Path to the HTML file
html_file_path = "3D_Visualization.html"

# Embed the HTML file in an iframe
st.components.v1.html(
    f"""
    <iframe src="{html_file_path}" width="100%" height="600px" style="border:none;"></iframe>
    """,
    height=600,
)
