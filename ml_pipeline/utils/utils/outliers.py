import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from eda import Eda
### New class 
class OutlierReplaceWithMedian(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, threshold=1.5):
        """
        Initialize the OutlierReplaceWithMedian with the columns to check for outliers
        and the IQR threshold for detecting them.
        
        Parameters:
        cols (list): List of column indices or names to check for outliers.
        threshold (float): Multiplier for the IQR to define outliers. Typically 1.5 or 3.
        """
        self.cols = cols
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self  # No fitting necessary for outlier removal
    def percentage_of_outliers(self, data, cols=None, threshold=1.5):
        """
        Calculate the percentage of outliers in each column.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        cols (list): List of column names or indices to check for outliers. If None, uses all numeric columns.
        threshold (float): Multiplier for the IQR to define outliers. Default is 1.5.

        Returns:
        pd.Series: A Series where the index is column names and the values are the percentage of outliers.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        if cols is None:
            cols = data.select_dtypes(include=[np.number]).columns
        
        outlier_percentages = {}
        for col in cols:
            if isinstance(col, int):
                col = data.columns[col]
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            percentage = len(outliers) / len(data) * 100
            outlier_percentages[col] = percentage
        
        return pd.Series(outlier_percentages)
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X)
        
        # If cols is None, use all numeric columns
        cols = self.cols if self.cols is not None else data.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if isinstance(col, int):  # If index is passed
                col = data.columns[col]  # Convert index to column name
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            
            median = data[col].median()
            data[col] = np.where(data[col] > upper_bound, median, data[col])
            data[col] = np.where(data[col] < lower_bound, median, data[col])
        
        return data.values  # Return as a NumPy array for compatibility with scikit-learn


use = Eda('Outcome',"mean","mode" )
df = use.read_file().head(10)

out=OutlierReplaceWithMedian()
df1 = out.percentage_of_outliers(df, cols=None, threshold=1.5)
print(df1)