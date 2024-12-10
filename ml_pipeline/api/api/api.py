from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from scaling_outliers import  OutlierReplaceWithMedian, ApplyScaling, CustomEncoder

# Load the trained model
model = joblib.load("random_new.pkl")
# Define the feature names (same as during training)
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request (expects JSON)
        data = request.json

        # Validate input format
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input format. Provide a 'features' key with a list of values."}), 400

        # Extract features
        features = data["features"]
        if len(features) != len(feature_names):
            return jsonify({"error": f"Expected {len(feature_names)} features, got {len(features)}."}), 400

        # Convert the features to a DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)

        # Make predictions
        predictions = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df).tolist()[0]

        # Return predictions as JSON
        return jsonify({
            "prediction": int(predictions),
            "probabilities": prediction_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)