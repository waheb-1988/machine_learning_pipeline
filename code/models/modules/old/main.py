import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from model import CM
app = Flask('mpg_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    vehicle = request.get_json()
    print('##############')
    print(type(vehicle))
    with open('C:\Abdelouaheb\perso\Ph\machine_learning_pipeline\code\models\model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    
    gg= CM(vehicle)
    predictions = gg.predict_mpg(vehicle, model)

    result = {
        'mpg_prediction': list(predictions)
    }
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"