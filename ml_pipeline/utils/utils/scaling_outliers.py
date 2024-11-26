from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class OutlierReplaceWithMedian(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.5):
        """
        Initializes the transformer for outlier handling.

        Parameters:
            threshold (float): Threshold for detecting outliers using the IQR method.
        """
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self  # No fitting necessary for this step
    
    def transform(self, X):
        """
        Replaces outliers in numeric columns with the median.

        Parameters:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with outliers replaced by median.
        """
        X = X.copy()
        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR

            # Replace outliers with the median
            median = X[col].median()
            X[col] = np.where((X[col] < lower_bound) | (X[col] > upper_bound), median, X[col])
        return X

class ApplyScaling(BaseEstimator, TransformerMixin):
    def __init__(self, scaling_technique="minmax"):
        """
        Initializes the transformer for scaling.

        Parameters:
            scaling_technique (str): Scaling technique to use ('standard', 'minmax', or 'robust').
        """
        self.scaling_technique = scaling_technique
        self.scaler = None

    def fit(self, X, y=None):
        # Choose the scaler
        if self.scaling_technique == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_technique == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling_technique == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling technique. Choose 'standard', 'minmax', or 'robust'.")

        # Fit the scaler on numeric columns
        self.scaler.fit(X.select_dtypes(include=["float64", "int64"]))
        return self

    def transform(self, X):
        """
        Scales numeric columns in the DataFrame.

        Parameters:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        X = X.copy()
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_dict=None, ordinal_categories=None):
        # Ensure the parameters are set correctly
        self.encoding_dict = encoding_dict if encoding_dict is not None else {}
        self.ordinal_categories = ordinal_categories if ordinal_categories is not None else {}

    def fit(self, X, y=None):
        # Fit method (no modification of internal parameters)
        return self

    def transform(self, X):
        # Transform method where you implement your encoding logic
        X_encoded = X.copy()
        # Apply encoding logic (example: replace categorical columns with values from encoding_dict)
        for col, mapping in self.encoding_dict.items():
            X_encoded[col] = X_encoded[col].map(mapping)
        return X_encoded

    def get_params(self, deep=True):
        # Ensure parameters can be cloned properly
        return {
            "encoding_dict": self.encoding_dict,
            "ordinal_categories": self.ordinal_categories
        }

    def set_params(self, **params):
        # Ensure the parameters are correctly set during grid search or cloning
        if "encoding_dict" in params:
            self.encoding_dict = params["encoding_dict"]
        if "ordinal_categories" in params:
            self.ordinal_categories = params["ordinal_categories"]
        return self

    def fit_transform(self, X, y=None):
        """
        Fits the encoders and transforms the data in one step.
        
        Parameters:
            X (pd.DataFrame): The input DataFrame.
            y (ignored): Compatibility with scikit-learn pipeline (not used).
        
        Returns:
            pd.DataFrame: Transformed DataFrame with encoded columns.
        """
        return self.fit(X, y).transform(X)

## Test
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from pathlib import Path
import os
dir_folder= Path.cwd().parent.parent
input_path = dir_folder/ "data" / "data" 
file_name = "diabetes.csv"
df= pd.read_csv(os.path.join(input_path,file_name))
x= df.drop(columns='Outcome')
print(x.shape)
y= df['Outcome']   
# ordinal_categories = {}    
# encoding_dict = {}    
# pipeline = Pipeline([
#     ("custom_encoding", CustomEncoder(encoding_dict=encoding_dict, ordinal_categories=ordinal_categories)),
#     ("outlier_replacement", OutlierReplaceWithMedian(threshold=1.5)),
#     ("scaling", ApplyScaling(scaling_technique="minmax")),
#     ("model", RandomForestClassifier(random_state=42))
# ])
# #


# x_t,x_te,y_t,y_te= train_test_split(x,y,test_size=.25,random_state=20, stratify=y)

# model = pipeline.fit(x_t,y_t)

# # Make Predictions
# y_pred_train = model.predict(x_t)
# y_pred_test = model.predict(x_te)


# # Calculate Accuracy 
# train_accuracy  = accuracy_score(y_pred_train,y_t)
# test_accuracy  = accuracy_score(y_pred_test,y_te)


# print(f"Training Accuracy: {train_accuracy:.2f}")
# print(f"Test Accuracy: {test_accuracy:.2f}")


# params = {
#     'model__n_estimators': [100, 200],
#     'model__max_depth': [None, 10, 20, 30],
#     'model__min_samples_split': [2, 5],
#     'model__min_samples_leaf': [1, 2, 4],
#     'model__bootstrap': [True, False]
# }
# nreg=GridSearchCV(pipeline,param_grid=params,cv=5, verbose=2, n_jobs=-1, scoring='accuracy')

# model1 = nreg.fit(x_t,y_t)



# print(model1.best_params_)
# print(model1.best_score_)
# # Make Predictions
# model2=model1.best_estimator_

# y_pred_train1 = model2.predict(x_t)
# y_pred_test1 = model2.predict(x_te)

# # Calculate Accuracy 
# train_accuracy1  = accuracy_score(y_pred_train1,y_t)
# test_accuracy1  = accuracy_score(y_pred_test1,y_te)


# print(f"Training Accuracy1: {train_accuracy1:.2f}")
# print(f"Test Accuracy1: {test_accuracy1:.2f}")

# import joblib
# joblib.dump(model2, "random_new.pkl")