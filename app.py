import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit app configuration
st.set_page_config(
    page_title="AmtariX",
    page_icon="🛰️",
    layout="wide"
)

# Function to create a horizontal navigation bar
def nav_bar():
    st.markdown(
        """
        <style>
        .nav-bar {
            background-color: #f8f9fa;
            padding: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
        }
        .nav-bar a {
            text-decoration: none;
            color: #007bff;
            font-size: 18px;
            font-weight: bold;
        }
        .nav-bar a:hover {
            text-decoration: underline;
            color: #0056b3;
        }
        </style>
        <div class="nav-bar">
            <a href="/?nav=app" target="_self">App</a>
            <a href="/?nav=globe" target="_self">Globe Visualization</a>
            <a href="/?nav=research" target="_self">Research Work</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Determine the current page from query parameters
query_params = st.experimental_get_query_params()
current_page = query_params.get("nav", ["app"])[0]

# Display navigation bar
nav_bar()

# Page content
if current_page == "app":
    # App content
    st.image("assets/banner.jpg", use_column_width=True)  # Display the banner at the top
    st.title("AmtariX")
    st.write("### 🛰️ Predicting Satellite Orbits with ML")
    
    st.write(
        """
        Welcome to AmtariX. This tool allows you to input TLE (Two-Line Element) data 
        to predict satellite orbits with precision.
        """
    )
    
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

elif current_page == "globe":
    # Globe Visualization content
    st.title("Globe Visualization")
    st.write("### 🗺️ Visualizing Satellite Orbits Globally")
    st.write(
        """
        This section is under development and will provide an interactive 3D globe to visualize satellite orbits.
        """
    )

elif current_page == "research":
    # Research Work content
    st.title("Research Work")
    st.write("### 📚 Detailed Research on Satellite Orbit Prediction")
    st.write(
        """
        Learn about the research and methodologies used for satellite orbit prediction, including the hybrid 
        modeling approach combining SGP4 and machine learning.
        """
    )
