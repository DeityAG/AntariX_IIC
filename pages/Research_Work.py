import streamlit as st

st.title("Research Work")
st.write("Project Implementation, Details, and References")

st.subheader("Overview of Implementation")
st.write("""
This project predicts satellite orbits using TLE data and an ML model. The predictions are visualized on a 3D globe 
using Plotly for an interactive experience. 
""")

st.subheader("References")
st.markdown("""
- **SGP4 Model for Orbit Prediction**: [Celestrak Documentation](https://celestrak.com/NORAD/documentation/)
- **Two-Line Element (TLE) Format**: [Wikipedia](https://en.wikipedia.org/wiki/Two-line_element_set)
- **Plotly for 3D Visualization**: [Plotly Documentation](https://plotly.com/python/)
""")

st.subheader("Tech Stack Used")
st.markdown("""
- **Python Libraries**: Streamlit, Plotly, Pandas, Matplotlib
- **Machine Learning**: Custom model for satellite orbit prediction
- **Data Sources**: Celestrak for TLE data
""")
