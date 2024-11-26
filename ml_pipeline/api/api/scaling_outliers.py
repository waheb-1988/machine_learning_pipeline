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

