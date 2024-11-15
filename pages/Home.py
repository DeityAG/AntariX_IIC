import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Logo and Title
st.title("Codeplay Satellite Orbit Predictor")
logo_path = "assets/logo.png"
st.image(logo_path, width=200)

# Workflow explanation
st.header("Overview")
st.write("""
This tool predicts satellite orbits using an ML model and visualizes paths on a 3D globe. 
It combines data processing, machine learning, and language model explanations.
""")

# Inputs for ML model (17 inputs)
st.header("ML Model Input")
input_data = []
for i in range(1, 18):
    value = st.number_input(f"Input {i}", value=0.0)
    input_data.append(value)

# Placeholder for ML model predictions
if st.button("Run Prediction"):
    # Placeholder for loading and using the ML model (e.g., a pickle file)
    st.write("Running ML Model...")
    # Replace with actual model predictions when available
    prediction = np.random.randn()  # Example dummy prediction
    st.write("## Prediction Output")
    st.write(f"Predicted Value: {prediction}")
    
    # Placeholder for ML model explanation from LLM
    st.write("## Model Explanation (from LLM)")
    explanation = "This is a sample explanation generated based on the ML model's output."
    st.write(explanation)

    # Example plot showing difference between dataset and ML model
    st.write("### Comparison Plot")
    actual_values = np.random.randn(10)  # Example actual values from dataset
    predicted_values = actual_values + np.random.normal(0, 0.1, size=10)  # Example ML model predictions

    # Plot the differences
    fig, ax = plt.subplots()
    ax.plot(actual_values, label="Actual Values")
    ax.plot(predicted_values, label="Predicted Values", linestyle="--")
    ax.set_title("Dataset vs ML Model Predictions")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Enter input values and click 'Run Prediction' to see results.")
