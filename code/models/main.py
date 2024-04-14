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

class CM():
    def __init__(self, df):
        self.df = df
        
    def custom_attr_adder(self,X, acc_on_power=True, acc_ix=4, cyl_ix=0, hpower_ix=2):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]  
      
##preprocess the Origin column in data
    def preprocess_origin_cols(self,data):
        df= pd.DataFrame(data)
        df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
        return df
    
    def num_pipeline_transformer(self, data):
        df= pd.DataFrame(data)
        numerics = ['float64', 'int64']

        num_attrs = df.select_dtypes(include=numerics)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attrs_adder', FunctionTransformer(self.custom_attr_adder, validate=False,
                                                kw_args={'acc_on_power': True, 'acc_ix': 4, 'cyl_ix': 0, 'hpower_ix': 2})),
            ('std_scaler', StandardScaler()),
        ])
        
        return num_attrs, num_pipeline

    def pipeline_transformer(self,data):
        '''
        Complete transformation pipeline for both
        nuerical and categorical data.
        
        Argument:
            data: original dataframe 
        Returns:
            prepared_data: transformed data, ready to use
        '''
        df= pd.DataFrame(data)
        cat_attrs = ["Origin"]
        num_attrs, num_pipeline = self.num_pipeline_transformer(df)
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, list(num_attrs)),
            
            ("cat", OneHotEncoder(), cat_attrs),
            ])
        prepared_data = full_pipeline.fit_transform(df)
        return prepared_data

    def predict_mpg(self,data, model):
    
        df = pd.DataFrame(data)
        preproc_df = self.preprocess_origin_cols(df)
        print(preproc_df)
        prepared_df = self.pipeline_transformer(preproc_df)
        #prepared_df = pipeline.transform(preproc_df)
        print(len(prepared_df[0]))
        y_pred = model.predict(prepared_df)
        return y_pred

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