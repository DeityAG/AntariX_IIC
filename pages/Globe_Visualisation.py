import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("3D Visualization")
st.write("Interactive 3D Satellite Orbit Visualization with Plotly")

# Define Earth's radius
earth_radius = 6371  # in km

# Generate coordinates for the Earth sphere
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
x = earth_radius * np.outer(np.cos(theta), np.sin(phi))
y = earth_radius * np.outer(np.sin(theta), np.sin(phi))
z = earth_radius * np.outer(np.ones(100), np.cos(phi))

# Define sample data for satellite path (replace with actual data)
satellite_longitudes = np.linspace(-120, 120, 50)  # longitude data for path
satellite_latitudes = np.linspace(-30, 30, 50)  # latitude data for path
satellite_altitude = earth_radius + 400  # altitude above Earth's surface in km

# Convert satellite path to Cartesian coordinates
sat_x = (satellite_altitude) * np.cos(np.radians(satellite_latitudes)) * np.cos(np.radians(satellite_longitudes))
sat_y = (satellite_altitude) * np.cos(np.radians(satellite_latitudes)) * np.sin(np.radians(satellite_longitudes))
sat_z = (satellite_altitude) * np.sin(np.radians(satellite_latitudes))

# Define sample data for predicted path (replace with actual data)
predicted_longitudes = np.linspace(-100, 100, 50)
predicted_latitudes = np.linspace(-20, 20, 50)
predicted_altitude = earth_radius + 420

# Convert predicted path to Cartesian coordinates
pred_x = (predicted_altitude) * np.cos(np.radians(predicted_latitudes)) * np.cos(np.radians(predicted_longitudes))
pred_y = (predicted_altitude) * np.cos(np.radians(predicted_latitudes)) * np.sin(np.radians(predicted_longitudes))
pred_z = (predicted_altitude) * np.sin(np.radians(predicted_latitudes))

# Create the plotly figure
fig = go.Figure()

# Add Earth's surface as a 3D surface
fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    colorscale="Earth",
    opacity=0.7,
    showscale=False
))

# Add satellite path as a red line
fig.add_trace(go.Scatter3d(
    x=sat_x, y=sat_y, z=sat_z,
    mode='lines+markers',
    line=dict(color='red', width=2),
    marker=dict(size=3, color='red'),
    name="Satellite Path"
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
    title="3D Satellite Orbit Visualization",
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
