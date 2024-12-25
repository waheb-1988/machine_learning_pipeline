import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import Marker
from streamlit_folium import folium_static
from geopy.distance import geodesic
import time
from streamlit_folium import st_folium

# Sample data generation
np.random.seed(42)
hoist_names = [f"Hoist {i}" for i in range(1, 5)]
well_names = [f"Well {i}" for i in range(1, 11)]
data = {
    "Hoist_name": np.random.choice(hoist_names, 40),
    "Well_name": np.random.choice(well_names, 40),
    "Date": pd.date_range(start="2024-01-01", periods=40),
    "lat": np.random.uniform(23.0, 24.0, 40),
    "long": np.random.uniform(56.0, 57.5, 40),
    "type": np.random.choice(["maintenance", "Abandonment", "Other"], 40)
}

df = pd.DataFrame(data)

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Marker colors based on type
type_colors = {
    "other": "blue",
    "deferment": "red",
    "maintenance": "green",
}

# Sidebar filters
st.sidebar.header("Filters")
default_hoist = df["Hoist_name"].unique()[0]
selected_hoist = st.sidebar.selectbox("Select Hoist", df["Hoist_name"].unique(), index=0)
selected_types = st.sidebar.multiselect("Select Types", df["type"].unique(), default=df["type"].unique())

# Button to update the map
if st.sidebar.button("Update Map"):
    st.session_state.filtered_data = df[
        (df["Hoist_name"] == selected_hoist) & (df["type"].isin(selected_types))
    ]

# Main Dashboard
st.title("Dynamic Movement Map")
st.write("Visualize movements of hoists and wells with connecting lines.")

# Create the map and initialize markers
if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
    filtered_data = st.session_state.filtered_data.sort_values("Date")

    # Create the folium map (centered around the mean location)
    m = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=8)

    # Add markers and tooltips based on selected data
    for _, row in filtered_data.iterrows():
        folium.Marker(
            location=[row["lat"], row["long"]],
            popup=(
                f"Hoist: {row['Hoist_name']}<br>"
                f"Well: {row['Well_name']}<br>"
                f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>"
                f"Type: {row['type']}"
            ),
            tooltip=f"{row['Well_name']} (Type: {row['type']})",
            icon=folium.Icon(color=type_colors.get(row["type"], "gray")),
        ).add_to(m)

    # Add polyline to show dynamic movement
    coordinates = []
    for i, coord in enumerate(filtered_data[["lat", "long"]].values):
        coordinates.append(coord.tolist())
        if len(coordinates) > 1:
            # Update the polyline dynamically as new coordinates are added
            folium.PolyLine(
                coordinates,
                color="orange",
                weight=2.5,
                opacity=0.8,
                tooltip=f"Movement path for {selected_hoist}",
            ).add_to(m)

    # Display map in the Streamlit app
    st_folium(m, width=800, height=500)

else:
    st.info("Select filters and click 'Update Map' to view the movement.")