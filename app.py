import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import statsmodels.api as sm

# Function to safely load the ARIMA model
@st.cache_resource
def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Function to process input data
def process_input_data(data_str):
    try:
        rows = [row.strip() for row in data_str.strip().split('\n') if row.strip()]
        data = []
        for row in rows:
            values = [float(x.strip()) for x in row.split(',')]
            data.append(values)
        return np.array(data)
    except Exception as e:
        st.error(f"Error processing input data: {str(e)}")
        return None

# Function to generate hybrid predictions
def hybrid_predictions(sgp4_positions, arima_model, days=30):
    try:
        # Calculate distances from SGP4 positions
        sgp4_distances = np.linalg.norm(sgp4_positions, axis=1)
        
        # Generate forecasted errors using ARIMA
        forecasted_errors = arima_model.forecast(steps=days)
        
        # Combine SGP4 predictions with forecasted errors
        hybrid_distances = sgp4_distances[:days] + forecasted_errors
        
        return hybrid_distances
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Function to plot predictions
def plot_predictions(true_positions, sgp4_positions, hybrid_positions, days):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate distances
        true_distances = np.linalg.norm(true_positions, axis=1)[:days]
        sgp4_distances = np.linalg.norm(sgp4_positions, axis=1)[:days]
        
        # Create plots
        ax.plot(range(days), true_distances, label="True Trajectory", color="green")
        ax.plot(range(days), sgp4_distances, label="SGP4 Prediction", color="blue", linestyle="--")
        ax.plot(range(days), hybrid_positions[:days], label="Hybrid Prediction", color="red", linestyle="-.")
        
        ax.set_title("Trajectory Predictions")
        ax.set_xlabel("Days")
        ax.set_ylabel("Distance (km)")
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Calculate and display metrics
        mse_sgp4 = np.mean((true_distances - sgp4_distances) ** 2)
        mse_hybrid = np.mean((true_distances - hybrid_positions[:days]) ** 2)
        improvement = ((mse_sgp4 - mse_hybrid) / mse_sgp4) * 100
        
        st.write(f"MSE (SGP4): {mse_sgp4:.2f}")
        st.write(f"MSE (Hybrid): {mse_hybrid:.2f}")
        st.write(f"Improvement: {improvement:.2f}%")
        
    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")

# Streamlit UI
st.set_page_config(page_title="Satellite Orbit Predictor", layout="wide")

st.title("🛰️ Satellite Orbit Predictor")
st.write("Input synthetic true positions and SGP4 predictions to forecast satellite orbit errors.")

# Load model once at startup
model = load_model("arima_model.pkl")

# Input sections
col1, col2 = st.columns(2)

with col1:
    true_positions_input = st.text_area(
        "Enter Synthetic True Positions (comma-separated x,y,z coordinates, one row per day)",
        help="Example: 1000,2000,3000\n1100,2100,3100"
    )

with col2:
    sgp4_positions_input = st.text_area(
        "Enter SGP4 Predictions (comma-separated x,y,z coordinates, one row per day)",
        help="Example: 1000,2000,3000\n1100,2100,3100"
    )

if st.button("Generate Predictions"):
    if model is None:
        st.error("Please ensure the ARIMA model file (arima_model.pkl) is present in the application directory.")
    else:
        # Process inputs
        true_positions = process_input_data(true_positions_input)
        sgp4_positions = process_input_data(sgp4_positions_input)
        
        if true_positions is not None and sgp4_positions is not None:
            if true_positions.shape != sgp4_positions.shape:
                st.error("Input shapes don't match. Please ensure both inputs have the same number of days and coordinates.")
            else:
                days = len(true_positions)
                hybrid_positions = hybrid_predictions(sgp4_positions, model, days)
                
                if hybrid_positions is not None:
                    st.success("Predictions generated successfully!")
                    plot_predictions(true_positions, sgp4_positions, hybrid_positions, days)

# Add information about input format
with st.expander("Input Format Help"):
    st.write("""
    ### Input Format
    - Each line represents one day of positions
    - Each line should contain 3 comma-separated values: x, y, z coordinates in kilometers
    - Example:
        ```
        1000,2000,3000
        1100,2100,3100
        1200,2200,3200
        ```
    """)
