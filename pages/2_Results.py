import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Prediction Results")

# Check if TLE data is available in session state
if 'tle_data' in st.session_state:
    tle_data = st.session_state['tle_data']
    st.write("### Input TLE Data")
    st.text(tle_data)  # Display TLE input data
    
    # Placeholder for loading and using a model (e.g., pickle file)
    # Assuming a dummy prediction result for demonstration
    predictions = {
        "Parameter": ["Predicted Altitude", "Predicted Velocity", "Orbital Inclination"],
        "Value": [550, 7.8, 53.0]  # Dummy values
    }
    df = pd.DataFrame(predictions)
    
    # Display numerical data
    st.write("### Prediction Results")
    st.dataframe(df)

    # Simple graph (for example purposes)
    st.write("### Graph of Predictions")
    fig = px.bar(df, x="Parameter", y="Value", title="Prediction Results")
    st.plotly_chart(fig)

else:
    st.warning("No TLE data available. Please go to the Welcome page and enter TLE data.")
