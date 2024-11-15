import streamlit as st
import matplotlib.pyplot as plt

st.title("Error Analysis")

st.write("Visualize error trends over different prediction horizons.")

# Placeholder data
days = [10, 20, 30]
sgp4_errors = [3.0, 6.5, 10.2]
hybrid_errors = [1.5, 2.8, 4.1]

# Plot error trends
plt.figure(figsize=(8, 6))
plt.plot(days, sgp4_errors, label="SGP4 Errors", marker="o")
plt.plot(days, hybrid_errors, label="Hybrid Model Errors", marker="o")
plt.xlabel("Prediction Horizon (Days)")
plt.ylabel("Error (km)")
plt.title("Error Reduction Trends")
plt.legend()
plt.grid(True)
st.pyplot(plt)
