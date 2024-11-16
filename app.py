import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
    Welcome to the Codeplay Satellite Orbit Predictor. Input synthetic true positions and SGP4 predictions to forecast satellite orbit errors.
    """
)

# Sidebar for additional functionalities
st.sidebar.header("Navigation")
st.sidebar.write("Navigate to different sections:")
nav_options = ["Input Data", "ARIMA Model Integration", "About the Model", "Contact"]
selected_option = st.sidebar.radio("Choose an option:", nav_options)

# Function to load the ARIMA model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    st.success(f"Model loaded from {filename}")
    return model

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

# Handle navigation for Input Data section
if selected_option == "Input Data":
    st.header("Input Synthetic True Positions and SGP4 Predictions")
    st.write(
        """
        Paste the synthetic true positions and SGP4 predictions below (comma-separated values).
        """
    )

    # Text areas for input
    true_positions_input = st.text_area(
        "Enter Synthetic True Positions (comma-separated for each row):",
        height=150,
        placeholder="Example:\n7212.65174,-2025.17796,1139.87170\n3052.87766,-624.96185,6897.97492\n..."
    )
    
    sgp4_positions_input = st.text_area(
        "Enter SGP4 Predictions (comma-separated for each row):",
        height=150,
        placeholder="Example:\n7004.25368,-1927.28567,2150.51236\n2122.34933,-340.74298,7255.98930\n..."
    )

    if st.button("Run Prediction"):
        if true_positions_input.strip() and sgp4_positions_input.strip():
            try:
                # Process input data
                true_positions = np.array([
                    list(map(float, row.split(',')))
                    for row in true_positions_input.strip().split('\n')
                ])
                sgp4_positions = np.array([
                    list(map(float, row.split(',')))
                    for row in sgp4_positions_input.strip().split('\n')
                ])

                st.write("### Processed Input Data")
                st.write("**Synthetic True Positions**")
                st.write(true_positions)
                st.write("**SGP4 Predictions**")
                st.write(sgp4_positions)

                # Load the ARIMA model
                model_path = "arima_model.pkl"  # Update with the correct path to your model
                loaded_model = load_model(model_path)

                # Generate hybrid predictions
                steps = len(sgp4_positions)
                hybrid_positions = hybrid_predictions(sgp4_positions, loaded_model, steps=steps)

                # Plot results
                st.write("### Prediction Results")
                plot_predictions(true_positions, sgp4_positions, hybrid_positions, days=steps)

            except Exception as e:
                st.error(f"An error occurred while processing the input data: {e}")
        else:
            st.error("Please provide both synthetic true positions and SGP4 predictions.")
    
    # Add TLE Data Table
    st.header("TLE Data Table")
    st.write(
        """
        This table explains the meaning of the different columns in a TLE (Two-Line Element) dataset.
        """
    )

    # Sample TLE table data
    table_data = [
        {"Column": 1, "Example": "1", "Description": "Line Number"},
        {"Column": "3-7", "Example": "25544", "Description": "Satellite Catalog Number"},
        {"Column": 8, "Example": "U", "Description": "Elset Classification"},
        {"Column": "10-17", "Example": "98067A", "Description": "International Designator"},
        {"Column": "19-32", "Example": "04236.56031392", "Description": "Element Set Epoch (UTC) *Note: spaces are acceptable in columns 21 & 22"},
        {"Column": "34-43", "Example": ".00020137", "Description": "1st Derivative of the Mean Motion with respect to Time"},
        {"Column": "45-52", "Example": "00000-0", "Description": "2nd Derivative of the Mean Motion with respect to Time (decimal point assumed)"},
        {"Column": "54-61", "Example": "16538-3", "Description": "B* Drag Term"},
        {"Column": 63, "Example": "0", "Description": "Element Set Type"},
        {"Column": "65-68", "Example": "999", "Description": "Element Number"},
        {"Column": 69, "Example": "3", "Description": "Checksum"},
    ]

    # Create DataFrame
    tle_df = pd.DataFrame(table_data)

    # Display table
    st.table(tle_df)

elif selected_option == "ARIMA Model Integration":
    st.header("ARIMA Model Integration")
    st.write(
        """
        Use a pre-trained ARIMA model to forecast satellite orbit errors. This section is integrated into the Input Data workflow.
        """
    )

elif selected_option == "About the Model":
    st.header("About the Model")
    st.write(
        """
        This tool uses a combination of the Simplified General Perturbations (SGP4) model and Machine Learning 
        to predict satellite orbits. Input synthetic true positions and SGP4 predictions for forecasting.
        """
    )

elif selected_option == "Contact":
    st.header("Contact")
    st.write(
        """
        For support or collaboration, please reach out to the development team.
        """
    )
