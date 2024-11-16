import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from datetime import datetime, timedelta

# Streamlit app configuration
st.set_page_config(
    page_title="Codeplay-Satellite Orbit Predictor",
    page_icon="🛰",
    layout="wide"
)

# Display the app title and logo
st.image("assets/banner.jpg", use_container_width=True)  
st.markdown(
    """
    <h1 style="font-size:36px;">
        Antari<span style="color:#29B5E8;">X</span>-Satellite Orbit Predictor
    </h1>
    """,
    unsafe_allow_html=True
)
st.write("### 🛰 Predicting Satellite Orbits with ML")

# Introduction text
st.write(
    """
    Welcome to the Codeplay Satellite Orbit Predictor. This tool allows you to input TLE (Two-Line Element) data 
    to predict satellite orbits with precision. Use the sections in the sidebar to navigate.
    """
)

# Sidebar for additional functionalities
st.sidebar.header("Navigation")
st.sidebar.write("Navigate to different sections:")
nav_options = ["Input TLE Data", "ARIMA Model Integration", "About the Model", "Contact"]
selected_option = st.sidebar.radio("Choose an option:", nav_options)

# Function to load the ARIMA model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    st.success(f"Model loaded from {filename}")
    return model

# Function to read and process TLE data
def read_tle_data(input_data):
    lines = input_data.strip().split("\n")
    satellites = []
    for i in range(0, len(lines), 2):  # Process two lines at a time
        satellite = {
            "name": f"Satellite {i // 2 + 1}",
            "line1": lines[i].strip(),
            "line2": lines[i + 1].strip()
        }
        satellites.append(satellite)
    return pd.DataFrame(satellites)

# Function to generate synthetic true positions
def generate_synthetic_truth(satellite_data, satellite_name, days=90):
    satellite_row = satellite_data[satellite_data['name'] == satellite_name].iloc[0]
    satellite = twoline2rv(satellite_row['line1'], satellite_row['line2'], wgs72)
    
    positions = []
    current_time = datetime.utcnow()
    
    for i in range(days):
        prediction_time = current_time + timedelta(days=i)
        position, _ = satellite.propagate(
            prediction_time.year,
            prediction_time.month,
            prediction_time.day,
            prediction_time.hour,
            prediction_time.minute,
            prediction_time.second
        )
        positions.append(np.array(position))
    
    positions = np.array(positions)
    
    # Add realistic perturbations
    atmospheric_drag = np.random.normal(0, 0.3, (days, 3))
    solar_radiation = 0.1 * np.sin(np.linspace(0, 2 * np.pi, days))[:, None]
    station_keeping = (np.random.randint(0, 2, size=(days, 3)) * 0.5)
    
    synthetic_positions = positions + atmospheric_drag + solar_radiation + station_keeping
    return synthetic_positions

# Function to compute SGP4 predictions
def compute_sgp4_predictions(satellite_data, satellite_name, days=30):
    satellite_row = satellite_data[satellite_data['name'] == satellite_name].iloc[0]
    satellite = twoline2rv(satellite_row['line1'], satellite_row['line2'], wgs72)
    
    sgp4_predictions = []
    current_time = datetime.utcnow()
    
    for i in range(days):
        prediction_time = current_time + timedelta(days=i)
        position, _ = satellite.propagate(
            prediction_time.year,
            prediction_time.month,
            prediction_time.day,
            prediction_time.hour,
            prediction_time.minute,
            prediction_time.second
        )
        sgp4_predictions.append(np.array(position))
    
    return np.array(sgp4_predictions)

# Function to generate hybrid predictions
def hybrid_predictions(sgp4_positions, arima_model, steps=30):
    forecasted_errors = arima_model.forecast(steps=steps)
    sgp4_distances = np.linalg.norm(sgp4_positions, axis=1)
    hybrid_distances = sgp4_distances[:steps] + forecasted_errors
    return hybrid_distances

# Function to plot predictions
def plot_predictions(true_positions, sgp4_positions, hybrid_positions, days):
    plt.figure(figsize=(10, 6))
    plt.plot(range(days), np.linalg.norm(true_positions, axis=1)[:days], label="True Trajectory", color="green")
    plt.plot(range(days), np.linalg.norm(sgp4_positions, axis=1)[:days], label="SGP4 Prediction", color="blue", linestyle="--")
    plt.plot(range(days), hybrid_positions, label="Hybrid Prediction", color="red", linestyle="-.")
    plt.title("Trajectory Predictions")
    plt.xlabel("Days")
    plt.ylabel("Distance (km)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Handle navigation for Input TLE Data section
if selected_option == "Input TLE Data":
    # TLE Data Input Section
    st.header("Input TLE Data")
    st.write("Paste or upload your TLE data below:")
    
    # Text area for TLE input
    tle_input = st.text_area(
        "Enter TLE data (multiple satellites supported):",
        placeholder="Paste your TLE data here...",
        height=200
    )
    
    # Process the TLE input and generate predictions
    if st.button("Run Prediction"):
        if tle_input.strip():
            st.write("Processing TLE data...")
            st.session_state['tle_data'] = tle_input.strip()

            # Step 1: Preprocess TLE Data
            satellite_data = read_tle_data(tle_input)
            st.write("Processed Satellite Data:")
            st.dataframe(satellite_data)
            
            # Step 2: Generate Predictions Dynamically
            model_path = "arima_model.pkl"  # Update this with the actual path to your model
            loaded_model = load_model(model_path)

            # Forecast using ARIMA
            steps = st.slider("Select Forecast Steps:", min_value=1, max_value=30, value=10, step=1)
            forecasted_errors = loaded_model.forecast(steps=steps)

            # Create a DataFrame for predictions
            predictions = {
                "Step": list(range(1, steps + 1)),
                "Forecasted Error": forecasted_errors
            }
            df = pd.DataFrame(predictions)

            # Display results
            st.header("Prediction Results")
            st.dataframe(df)

            # Visualize results
            st.write("### Graph of Predictions")
            fig = px.line(
                df,
                x="Step",
                y="Forecasted Error",
                title="Forecasted Error per Step",
                labels={"Forecasted Error": "Error", "Step": "Prediction Step"}
            )
            st.plotly_chart(fig)

            # Display TLE Data Table
            st.header("TLE Data Table")
            tle_table_data = [
                {"Column": 1, "Example": "1", "Description": "Line Number"},
                {"Column": "3-7", "Example": "25544", "Description": "Satellite Catalog Number"},
                {"Column": 8, "Example": "U", "Description": "Elset Classification"},
                {"Column": "10-17", "Example": "98067A", "Description": "International Designator"},
                {"Column": "19-32", "Example": "04236.56031392", "Description": "Element Set Epoch (UTC)"},
                {"Column": "34-43", "Example": ".00020137", "Description": "1st Derivative of the Mean Motion"},
                {"Column": "45-52", "Example": "00000-0", "Description": "2nd Derivative of the Mean Motion"},
                {"Column": "54-61", "Example": "16538-3", "Description": "B* Drag Term"},
                {"Column": 63, "Example": "0", "Description": "Element Set Type"},
                {"Column": "65-68", "Example": "999", "Description": "Element Number"},
                {"Column": 69, "Example": "3", "Description": "Checksum"},
            ]
            tle_df = pd.DataFrame(tle_table_data)
            st.table(tle_df)
        else:
            st.error("Please input valid TLE data before running predictions.")

elif selected_option == "ARIMA Model Integration":
    st.header("ARIMA Model Integration")
    st.write("This section uses a pre-trained ARIMA model to forecast satellite orbit errors.")
    # (Remaining code for this section...)

elif selected_option == "About the Model":
    st.header("About the Model")
    st.write(
        """
        This tool uses a combination of the Simplified General Perturbations (SGP4) model and Machine Learning 
        to predict satellite orbits. You can paste TLE data to visualize predictions.
        """
    )
