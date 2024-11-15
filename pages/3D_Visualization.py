import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("3D Visualization")
st.write("Interactive 3D Satellite Orbit Visualization with Earth Texture and Predicted Path")

# Define Earth's radius
earth_radius = 6371  # in km

# Load Earth texture as colorscale for the sphere
earth_texture_url = "https://raw.githubusercontent.com/plotly/datasets/master/earth_cmap.jpg"

# Generate spherical coordinates for Earth surface
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
x = earth_radius * np.outer(np.cos(theta), np.sin(phi))
y = earth_radius * np.outer(np.sin(theta), np.sin(phi))
z = earth_radius * np.outer(np.ones(100), np.cos(phi))

# Satellite TLE data (sample)
mean_anomaly = 255.592      # degrees
inclination = 53.0523       # degrees
right_ascension = 181.7416  # degrees

# Generate sample satellite path based on TLE (assuming circular orbit for simplicity)
satellite_longitudes = np.linspace(-180, 180, 50)
satellite_latitudes = inclination * np.sin(np.radians(satellite_longitudes))
satellite_altitude = earth_radius + 400  # altitude in km

# Convert satellite path to Cartesian coordinates
sat_x = (satellite_altitude) * np.cos(np.radians(satellite_latitudes)) * np.cos(np.radians(satellite_longitudes))
sat_y = (satellite_altitude) * np.cos(np.radians(satellite_latitudes)) * np.sin(np.radians(satellite_longitudes))
sat_z = (satellite_altitude) * np.sin(np.radians(satellite_latitudes))

# Simulated ML model prediction for deviation (sample synthetic data)
predicted_longitudes = np.linspace(-180, 180, 50)
predicted_latitudes = inclination * np.sin(np.radians(predicted_longitudes) + 0.1)  # Slight deviation
predicted_altitude = earth_radius + 420  # slightly different altitude

# Convert predicted path to Cartesian coordinates
pred_x = (predicted_altitude) * np.cos(np.radians(predicted_latitudes)) * np.cos(np.radians(predicted_longitudes))
pred_y = (predicted_altitude) * np.cos(np.radians(predicted_latitudes)) * np.sin(np.radians(predicted_longitudes))
pred_z = (predicted_altitude) * np.sin(np.radians(predicted_latitudes))

# Create the Plotly 3D figure
fig = go.Figure()

# Add Earth's surface with texture
fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    surfacecolor=np.flipud(np.arange(z.shape[1])[None, :] * np.ones_like(z)),
    colorscale=[[0, 'rgb(0, 0, 255)'], [0.5, 'rgb(255, 255, 255)'], [1, 'rgb(0, 100, 0)']],
    opacity=0.7,
    cmin=0,
    cmax=2 * np.pi,
    showscale=False
))

# Add satellite path as a red line
fig.add_trace(go.Scatter3d(
    x=sat_x, y=sat_y, z=sat_z,
    mode='lines+markers',
    line=dict(color='red', width=2),
    marker=dict(size=3, color='red'),
    name="Expected Satellite Path"
))

# Add predicted path as a blue line
fig.add_trace(go.Scatter3d(
    x=pred_x, y=pred_y, z=pred_z,
    mode='lines+markers',
    line=dict(color='blue', width=2),
    marker=dict(size=3, color='blue'),
    name="Predicted Path"
))

# Update layout for 3D visualization
fig.update_layout(
    title="3D Satellite Orbit Visualization with Earth Texture",
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
    showlegend=True,
    height=700,
    margin=dict(t=0, b=0, l=0, r=0)
)

# Show the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)
