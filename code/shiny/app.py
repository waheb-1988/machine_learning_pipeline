import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import Marker
from streamlit_folium import folium_static
from geopy.distance import geodesic
import time

# Sample data generation
np.random.seed(42)
hoist_names = [f"Hoist {i}" for i in range(1, 5)]
well_names = [f"Well {i}" for i in range(1, 11)]
df = {
    "Hoist_name": np.random.choice(hoist_names, 40),
    "Well_name": np.random.choice(well_names, 40),
    "Date": pd.date_range(start="2024-01-01", periods=40),
    "lat": np.random.uniform(23.0, 24.0, 40),
    "long": np.random.uniform(56.0, 57.5, 40),
    "type": np.random.choice(["maintenance", "deferment", "other"], 40)
}

data = pd.DataFrame(df)

# Sidebar for filters
st.sidebar.header("Filters")
hoist_filter = st.sidebar.selectbox("Select Hoist:", options=data['Hoist_name'].unique())
type_filter = st.sidebar.selectbox("Select Type:", options=data['type'].unique())

# Filter data based on selections
filtered_data = data[(data['Hoist_name'] == hoist_filter) & (data['type'] == type_filter)]

# Create a Folium map centered at the first point with adjusted zoom level
m = folium.Map(location=[filtered_data['lat'].iloc[0], filtered_data['long'].iloc[0]], zoom_start=10, width="100%", height="600px")

# Add markers and animated polyline step-by-step
for i in range(len(filtered_data) - 1):
    # Current and next point
    current = (filtered_data.iloc[i]['lat'], filtered_data.iloc[i]['long'])
    next_point = (filtered_data.iloc[i + 1]['lat'], filtered_data.iloc[i + 1]['long'])

    # Add marker at the current point
    folium.Marker(current, popup=f"{filtered_data.iloc[i]['Well_name']}").add_to(m)

    # Add polyline from current to next with animation step
    folium.PolyLine([current, next_point], color="blue", weight=2.5, opacity=1).add_to(m)
    
    # Delay to simulate movement from one point to the next
    time.sleep(0.5)  # Adjust the sleep time to control animation speed

# Display the map
st.header("Animated Movement Map")
folium_static(m)  # Use folium_static to render the map in Streamlit
