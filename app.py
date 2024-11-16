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

# ARIMA Model Integration Section
if selected_option == "ARIMA Model Integration":
    st.header("ARIMA Model Integration")
    st.write("This section uses a pre-trained ARIMA model to forecast satellite orbit errors.")

    # Use the pre-defined model path
    model_path = "arima_model.pkl"  # The model is in your repository
    try:
        # Load the ARIMA model
        loaded_model = load_model(model_path)

        # Interactive slider for forecast steps
        steps = st.slider("Select Forecast Steps:", min_value=1, max_value=30, value=10, step=1)

        # SGP4 positions input
        sgp4_positions = np.random.rand(100, 3) * 1000  # Replace with real data
        true_positions = sgp4_positions + np.random.normal(0, 50, sgp4_positions.shape)  # Example synthetic truth

        # Generate hybrid predictions
        hybrid_positions = hybrid_predictions(sgp4_positions, loaded_model, steps=steps)

        # Display prediction results
        st.write(f"### Hybrid Predictions for {steps} Steps:")
        st.write(hybrid_positions[:steps])

        # Plot the predictions
        st.subheader("Prediction Plot")
        plot_predictions(true_positions, sgp4_positions, hybrid_positions, steps)

    except FileNotFoundError:
        st.error("The ARIMA model file 'arima_model.pkl' was not found. Please ensure the file exists.")

# Handle navigation for Input TLE Data section
elif selected_option == "Input TLE Data":
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

            # Load the ARIMA model
            model_path = "arima_model.pkl"  # Update this with the actual path to your model
            loaded_model = load_model(model_path)

            # Generate predictions dynamically
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
        else:
            st.error("Please input or upload TLE data before running predictions.")
    
    # Add TLE Data Table directly below input section
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

elif selected_option == "About the Model":
    st.header("About the Model")
    st.write(
        """
        This tool uses a combination of the Simplified General Perturbations (SGP4) model and Machine Learning 
        to predict satellite orbits. You can paste TLE data to visualize predictions.
        """
    )

elif selected_option == "Contact":
    st.header("Contact")
    st.write(
        """
        For support or collaboration, please reach out to the development team.
        """
    )
