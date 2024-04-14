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
        
    def custom_attr_adder(self, X, acc_on_power=True, acc_ix=4, cyl_ix=0, hpower_ix=2):
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Reshape if only one row
        
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]
        
        # Ensure the result array always has exactly 11 features
        if result.shape[1] < 11:
            zeros_to_add = np.zeros((result.shape[0], 11 - result.shape[1]))
            result = np.hstack((result, zeros_to_add))
        elif result.shape[1] > 11:
            result = result[:, :-1]  # Remove extra feature
        
        return result[:, :11]  # Ensure only 11 features are returned
##preprocess the Origin column in data
    def preprocess_origin_cols(self,df):
        df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
        return df
    
    def num_pipeline_transformer(self, data):
        numerics = ['float64', 'int64']

        num_attrs = data.select_dtypes(include=numerics)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attrs_adder', FunctionTransformer(self.custom_attr_adder,validate=False,kw_args={'acc_on_power': True, 'acc_ix': 4, 'cyl_ix': 0, 'hpower_ix': 2})),
                                            
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
        cat_attrs = ["Origin"]
        num_attrs, num_pipeline = self.num_pipeline_transformer(data)
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, list(num_attrs)),
            
            ("cat", OneHotEncoder(), cat_attrs),
            ])
        prepared_data = full_pipeline.fit_transform(data)
        return prepared_data

    def predict_mpg(self,config, model):
    
        
        df = pd.DataFrame(config)
        
        
        preproc_df = self.preprocess_origin_cols(df)
        print(preproc_df)
        prepared_df = self.pipeline_transformer(preproc_df)
        #prepared_df = pipeline.transform(preproc_df)
        print(len(prepared_df[0]))
        y_pred = model.predict(prepared_df)
        return y_pred
    
    
    
############## Test
# gg={
#     "Cylinders": [4, 6, 8],
#     "Displacement": [155.0, 160.0, 165.5],
#     "Horsepower": [93.0, 130.0, 98.0],
#     "Weight": [2500.0, 3150.0, 2600.0],
#     "Acceleration": [15.0, 14.0, 16.0],
#     "Model Year": [81, 80, 78],
#     "Origin": [3, 2, 1]
# }

# mm=pd.DataFrame(gg)
# import pickle
# with open('C:\Abdelouaheb\perso\Ph\machine_learning_pipeline\code\models\model.bin', 'rb') as f_in:
#     model = pickle.load(f_in)
# print(mm)
# ff=CM(mm)

# data = ff.predict_mpg(mm,model)


# print(data)

