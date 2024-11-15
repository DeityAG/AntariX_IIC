import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Logo and title
st.title("Codeplay Satellite Orbit Predictor")
logo_path = "assets/logo.png"
if logo_path:
    st.image(logo_path, width=200)

# Basic introduction text
st.subheader("Welcome to Codeplay Satellite Orbit Predictor")
st.write("""
    This application allows you to input satellite telemetry data and generate predictions 
    using an ML model. Based on the predicted values, we also provide an interactive globe visualization 
    and an explanation generated from a language model.
""")

# Input fields for ML model
st.subheader("Input Data")
st.write("Please enter the following parameters for the satellite (17 inputs):")
input_data = {}
for i in range(1, 18):
    input_data[f'Input_{i}'] = st.text_input(f'Input {i}', '')

# Run prediction button (placeholder for ML model processing)
if st.button("Run Prediction"):
    # Placeholder for ML model integration
    st.write("Running the prediction model...")  
    # For now, we display mock predictions
    predictions = {f"Predicted Output {i}": np.random.rand() for i in range(1, 6)}
    st.write("### Model Predictions")
    st.json(predictions)  # Display predictions as JSON for now

    # Placeholder for difference plots between dataset and predictions
    st.subheader("Difference Plots")
    dataset_values = [np.random.rand() * 10 for _ in range(5)]
    predicted_values = list(predictions.values())
    diff_df = pd.DataFrame({
        "Parameter": [f"Feature {i}" for i in range(1, 6)],
        "Dataset Values": dataset_values,
        "Predicted Values": predicted_values
    })

    fig = px.bar(diff_df, x="Parameter", y=["Dataset Values", "Predicted Values"], barmode="group")
    st.plotly_chart(fig)

    # Placeholder for LLM explanation based on predictions
    st.subheader("LLM Explanation")
    st.write("Based on the predicted values, here is a contextual explanation:")
    st.write("LLM output: This feature is under development and will provide explanations based on model predictions.")
else:
    st.info("Enter the input data and click 'Run Prediction' to view results.")
