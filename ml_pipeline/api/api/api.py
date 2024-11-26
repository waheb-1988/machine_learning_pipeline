from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from scaling_outliers import  OutlierReplaceWithMedian, ApplyScaling, CustomEncoder

model = joblib.load("random_new.pkl")

# Load the trained model
#model = pickle.load(open("logitic1.pkl", "rb"))

app = Flask(__name__)

# Define the feature names (same as during training)
feature_names = [
    "Pregnancies","Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
    "DiabetesPedigreeFunction", "Age"
]

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request (expects JSON)
    data = request.json
    
    try:
        # Convert the input data into a pandas DataFrame
        input_df = pd.DataFrame(data)
        
        # Ensure the columns match the expected feature names
        input_df = input_df[feature_names]
        
        # Convert features to numpy array and scale if necessary
        #features = scaler.transform(input_df.values)  # Assuming scaling is applied
        
        # Make predictions for each set of features
        predictions = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Return the predictions as JSON
        result = []
        for i in range(len(predictions)):
            result.append({
                "prediction": int(predictions[i]),
                "probabilities": prediction_proba[i].tolist()  # Convert to list for JSON serialization
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)