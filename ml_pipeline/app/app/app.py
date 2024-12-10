import streamlit as st
import requests

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"  # Update if your API runs on a different URL/port

# Streamlit app title
st.title("Logistic Regression Prediction App")

# Input fields for features
st.header("Enter the Features for Prediction")
feature_1 = st.number_input("Pregnancies", value=0, step=1)
feature_2 = st.number_input("Glucose", value=0.0)
feature_3 = st.number_input("Blood Pressure", value=0.0)
feature_4 = st.number_input("Skin Thickness", value=0.0)
feature_5 = st.number_input("Insulin", value=0.0)
feature_6 = st.number_input("BMI", value=0.0)
feature_7 = st.number_input("Diabetes Pedigree Function", value=0.0)
feature_8 = st.number_input("Age", value=0, step=1)

# Collect features into a list
features = [feature_1, feature_2, feature_3, feature_4, feature_5, 
            feature_6, feature_7, feature_8]

# Predict button
if st.button("Get Prediction"):
    try:
        # Prepare the payload
        payload = {"features": features}

        # Send the request to the Flask API
        response = requests.post(API_URL, json=payload)

        # Handle response
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Probabilities: {result['probabilities']}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
