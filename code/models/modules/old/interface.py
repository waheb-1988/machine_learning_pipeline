import streamlit as st
import requests

# Streamlit UI
st.title("Car MPG Prediction")
cylinders_values = [st.number_input(f"Cylinders {i}", min_value=3, max_value=8, step=1, value=3) for i in range(1, 4)]
displacement_values = [st.number_input(f"Displacement {i}", min_value=1, value=1) for i in range(1, 4)]
horsepower_values = [st.number_input(f"Horsepower {i}", min_value=1, value=1) for i in range(1, 4)]
weight_values = [st.number_input(f"Weight {i}", min_value=1, value=1) for i in range(1, 4)]
acceleration_values = [st.number_input(f"Acceleration {i}", min_value=1, value=1) for i in range(1, 4)]
model_year_values = [st.number_input(f"Model Year {i}", min_value=70, max_value=99, step=1, value=70) for i in range(1, 4)]
origin_values = [st.number_input(f"Origin {i}", min_value=1, max_value=3, step=1, value=1) for i in range(1, 4)]

# Combine values into a single dictionary
input_data = {
    "Cylinders": cylinders_values,
    "Displacement": displacement_values,
    "Horsepower": horsepower_values,
    "Weight": weight_values,
    "Acceleration": acceleration_values,
    "Model Year": model_year_values,
    "Origin": origin_values
}

print("input_data")
print(input_data)
# Button to trigger prediction
if st.button("Predict MPG"):
    st.json(input_data)

    # Send data to API for prediction
    response = requests.post("http://localhost:5000/predict", json=input_data)  # Update endpoint URL
    
    # Display prediction result
    if response.status_code == 200:
        prediction = response.json()["mpg_prediction"]
        st.write("MPG Prediction:", prediction)
    else:
        st.write("Error:", response.text)
