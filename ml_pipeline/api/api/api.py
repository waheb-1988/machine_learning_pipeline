from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from custom_tr import OutlierReplaceWithMedian  # Include or import your custom class

model = joblib.load("logitic1.pkl")

# Load the trained model
#model = pickle.load(open("logitic1.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request (expects JSON)
    data = request.json
    print("heheheh")
    print(data)
    try:
        # Extract features from the request
        features = np.array(data['features']).reshape(1, -1)  # Reshape for a single prediction
        
        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        
        # Return the prediction as JSON
        return jsonify({
            "prediction": int(prediction[0]),
            "probabilities": prediction_proba.tolist()[0]  # Convert to list for JSON serialization
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
