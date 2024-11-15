import streamlit as st
import plotly.graph_objects as go

st.title("3D Visualization")

# Simulated 3D orbit data
fig = go.Figure(data=[
    go.Scatter3d(
        x=[1, 2, 3],
        y=[1, 2, 3],
        z=[1, 2, 3],
        mode='lines',
        marker=dict(size=5)
    )
])

fig.update_layout(title="3D Satellite Orbit", scene=dict(
    xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"
))

st.plotly_chart(fig, use_container_width=True)
