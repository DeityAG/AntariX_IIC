import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Set up page
st.title("3D Visualization")
st.write("Interactive 3D Satellite Orbit Visualization with Plotly")

# Earth surface texture coordinates
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
x = 6371 * np.sin(phi) * np.cos(theta)
y = 6371 * np.sin(phi) * np.sin(theta)
z = 6371 * np.cos(phi)

# Generate sample satellite track coordinates (use real data in actual implementation)
# These are placeholders; replace with your dataset's coordinates
satellite_longitude = np.linspace(-180, 180, 100)
satellite_latitude = 20 * np.sin(np.radians(satellite_longitude))
satellite_altitude = 700  # Example altitude in kilometers

# Convert lat/lon/alt to Cartesian coordinates for plotting
sat_x = (6371 + satellite_altitude) * np.cos(np.radians(satellite_latitude)) * np.cos(np.radians(satellite_longitude))
sat_y = (6371 + satellite_altitude) * np.cos(np.radians(satellite_latitude)) * np.sin(np.radians(satellite_longitude))
sat_z = (6371 + satellite_altitude) * np.sin(np.radians(satellite_latitude))

# Generate sample predicted path coordinates (deviated path)
predicted_longitude = np.linspace(-180, 180, 100)
predicted_latitude = 20 * np.sin(np.radians(predicted_longitude) + 0.1)  # Offset slightly for deviation
predicted_altitude = 705  # Slightly different altitude

# Convert predicted lat/lon/alt to Cartesian coordinates
pred_x = (6371 + predicted_altitude) * np.cos(np.radians(predicted_latitude)) * np.cos(np.radians(predicted_longitude))
pred_y = (6371 + predicted_altitude) * np.cos(np.radians(predicted_latitude)) * np.sin(np.radians(predicted_longitude))
pred_z = (6371 + predicted_altitude) * np.sin(np.radians(predicted_latitude))

# Create Plotly 3D figure
fig = go.Figure()

# Plot Earth surface
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.5))

# Plot satellite actual path
fig.add_trace(go.Scatter3d(
    x=sat_x, y=sat_y, z=sat_z,
    mode="lines",
    line=dict(color="red", width=2),
    name="Actual Satellite Path"
))

# Plot predicted (deviated) path
fig.add_trace(go.Scatter3d(
    x=pred_x, y=pred_y, z=pred_z,
    mode="lines",
    line=dict(color="green", width=2, dash="dash"),
    name="Predicted Path"
))

# Configure layout for better appearance
fig.update_layout(
    scene=dict(
        xaxis=dict(title="X (km)", showgrid=False, zeroline=False, visible=False),
        yaxis=dict(title="Y (km)", showgrid=False, zeroline=False, visible=False),
        zaxis=dict(title="Z (km)", showgrid=False, zeroline=False, visible=False),
        aspectmode="data"
    ),
    title="3D Satellite Orbit Visualization",
    showlegend=True
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
