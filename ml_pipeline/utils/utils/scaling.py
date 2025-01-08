from eda import Eda
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class ApplyScaling(BaseEstimator, TransformerMixin):
    def __init__(self, scaling_technique=None):
        """
        Initializes the transformer for scaling.

        Parameters:
            scaling_technique (str or None): Scaling technique to use ('standard', 'minmax', or 'robust').
                                             If None, no scaling is applied.
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
        elif self.scaling_technique is None:
            self.scaler = None
            return self  # No fitting required
        else:
            raise ValueError("Invalid scaling technique. Choose 'standard', 'minmax', 'robust', or None.")

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
        if self.scaler is None:
            return X  # Return the input as-is if no scaling technique is specified

        X = X.copy()
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X

# if __name__ == '__main__':
    
    
#     use = Eda('Outcome',"mean","mode" )
#     df = use.read_file().head(10)
#     print(df.info())
#     sca= ApplyScaling(scaling_technique = "standard")

#     df1= sca.fit_transform(df)

#     print(df1)

        
    
        




