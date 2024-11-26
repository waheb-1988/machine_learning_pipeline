# Class / method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pathlib 
import os
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
class Eda :
    def __init__(self,target,technique_inpute_numeric,technique_inpute_categorial):
        self.target = target
        self.technique_inpute_numeric = technique_inpute_numeric 
        self.technique_inpute_categorial = technique_inpute_categorial
        
      
    @staticmethod    
    def read_file():
        dir_folder= pathlib.Path.cwd().parent.parent
        input_path = dir_folder/ "data" / "data" 
        file_name = "diabetes_missing_values.csv"
        df= pd.read_csv(os.path.join(input_path,file_name))
        return df
    
    @staticmethod
    def data_diagnostic(df):
        print("#"*50)
        print(df.info())
        print("#"*50)
        print("The number of total rows  {x: .0f} ".format(x=df.shape[0]))
        print("The number of total variables {x: .0f} ".format(x=df.shape[1]))
        print("The variables names {x:} ".format(x=list(df.columns.values)))

        column_headers =list(df.columns.values)
        qualitative_columns = [col for col in column_headers if df[col].dtype=="object"]
        quantitative_columns = [col for col in column_headers if df[col].dtype in ['int64', 'float64']]

        print("The qualitative variables {x:} ".format(x=qualitative_columns))
        print("The quantitative variables {x:} ".format(x=quantitative_columns))
        print("#"*50)
        print("Total number missing value {x:} ".format(x=df.isnull().sum()))

    @staticmethod
    def numeric_analysis(df):
        
    
        return print(df.describe())
    
    
    def target_variable_balance_check(self, df, target):
        # Calculate percentage distribution
        value_counts = df[target].value_counts(normalize=True) * 100
        
        # Print value counts to verify
        print("Target Variable Percentage Distribution:")
        print(value_counts)
        
        # Plot percentage distribution
        sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
        plt.title(f'Distribution of {target} (in %)')
        plt.xlabel('Target Variable')
        plt.ylabel('Percentage (%)')
        plt.show()
        
        # Check balance
        threshold = 10  # Example threshold for imbalance (adjust as needed)
        max_diff = value_counts.max() - value_counts.min()
        if max_diff > threshold:
            print("The data is unbalanced.")
        else:
            print("The data is balanced.")
            
    @staticmethod
    def univariate_analysis(df, base_folder="univariate_analysis"):
        # Ensure the base folder exists
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Create a folder for the analysis
            col_folder = os.path.join(base_folder, col)
            if not os.path.exists(col_folder):
                os.makedirs(col_folder)
            
            print(f"\nPerforming Univariate Analysis for: {col}")
            
            # Create a single figure with 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Univariate Analysis for {col}', fontsize=16)
            
            # Bar Chart
            sns.barplot(
                x=df[col].value_counts().index, 
                y=df[col].value_counts().values, 
                palette="viridis", 
                ax=axes[0, 0]
            )
            axes[0, 0].set_title('Bar Chart')
            axes[0, 0].set_xlabel(col)
            axes[0, 0].set_ylabel('Frequency')
            
            # Box Plot
            sns.boxplot(y=df[col], palette="viridis", ax=axes[0, 1])
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].set_xlabel(col)
            
            # Density Plot
            sns.kdeplot(df[col], fill=True, color="blue", alpha=0.6, ax=axes[1, 0])
            axes[1, 0].set_title('Density Plot')
            axes[1, 0].set_xlabel(col)
            axes[1, 0].set_ylabel('Density')
            
            # Histogram
            sns.histplot(df[col], kde=False, color="green", ax=axes[1, 1])
            axes[1, 1].set_title('Histogram')
            axes[1, 1].set_xlabel(col)
            axes[1, 1].set_ylabel('Frequency')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the main title
            
            # Save the combined plot
            combined_plot_path = os.path.join(col_folder, f"{col}_univariate_analysis.png")
            plt.savefig(combined_plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Combined plots for {col} saved in: {combined_plot_path}")

    # def analysis of categorial
    def multyvarie_analysis(self, df,base_folder="multivariate_analysis"):
        """
        Performs pairwise plots for multivariate analysis with respect to the target variable.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to analyze.
            target (str): The target variable for the hue in the pairplot.
            base_folder (str): Directory to save the pairplot images.
        """
        # Ensure the base folder exists
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # Generate the pairplot
        pairplot_path = os.path.join(base_folder, f"pairplot_{self.target}.png")
        sns.pairplot(df, hue=self.target, diag_kind='kde', height=3)
        plt.savefig(pairplot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Pairplot saved in: {pairplot_path}")
    
    

    def input_missing_values(self, df, base_folder="missing_values_analysis"):
        """
        Summarizes missing values in the DataFrame, provides imputation options for missing data,
        and generates a heatmap of missing values. The heatmap is saved in the specified directory.

        Parameters:
            df (pd.DataFrame): The DataFrame to analyze and impute.
            base_folder (str): Directory to save the heatmap image.

        Returns:
            pd.DataFrame: A summary of missing values with variable name, number of missing values,
                        percentage of missing, and variable type.
            pd.DataFrame: The DataFrame with imputed missing values.
        """
        
        # 1. Summarize missing values
        missing_summary = pd.DataFrame({
            'Variable Name': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing Percentage': (df.isnull().mean().values * 100),
            'Variable Type': df.dtypes.values
        })
        df_missing_summary = missing_summary.copy()
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
        
        print("Missing Values Summary:")
        print(df_missing_summary)

        # 2. Ensure the base folder exists for saving the heatmap
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # 3. Generate heatmap of missing values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, xticklabels=df.columns)
        plt.title("Missing Values Heatmap", fontsize=16)
        
        # Save the heatmap in the specified folder with a default name
        heatmap_path = os.path.join(base_folder, "missing_values_heatmap.png")
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        print(f"Missing values heatmap saved to: {heatmap_path}")
        
        # 4. Impute missing values for each column
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:  # Numeric columns
                    print(f"\nHandling missing values for numeric column: {col}")
                    # Choose imputation technique for numeric data
                    technique = self.technique_inpute_numeric
                    
                    if technique == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif technique == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif technique == 'min':
                        df[col].fillna(df[col].min(), inplace=True)
                    elif technique == 'max':
                        df[col].fillna(df[col].max(), inplace=True)
                    elif technique == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        print(f"Invalid or unsupported technique for {col}. Skipping imputation.")
                
                elif df[col].dtype == 'object':  # Categorical columns
                    print(f"\nHandling missing values for categorical column: {col}")
                    # Choose imputation technique for categorical data
                    technique_cat = self.technique_inpute_categorial.lower()
                    if technique_cat == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif technique_cat == 'most_frequent':
                        most_frequent = df[col].value_counts().idxmax()
                        df[col].fillna(most_frequent, inplace=True)
                    elif technique_cat == 'constant_value':
                        constant_value = input(f"Enter constant value to impute for {col}: ")
                        df[col].fillna(constant_value, inplace=True)
                    elif technique_cat == 'random_value':
                        unique_values = df[col].dropna().unique()
                        random_value = random.choice(unique_values)
                        df[col].fillna(random_value, inplace=True)
                    else:
                        print(f"Invalid or unsupported technique for {col}. Skipping imputation.")
            
        
        return df_missing_summary, df
    
    @staticmethod
    def apply_scaling(df, base_folder="scaling_analysis", scaling_technique="standard"):
        """
        Standardizes or scales numeric columns in the DataFrame based on the specified technique.

        Parameters:
            df (pd.DataFrame): The input DataFrame to scale.
            base_folder (str): Directory to save the summary and visualization of scaling.
            scaling_technique (str): The scaling technique to apply. Options are:
                                    'standard' (StandardScaler),
                                    'minmax' (MinMaxScaler),
                                    'robust' (RobustScaler).

        Returns:
            pd.DataFrame: A summary of the scaling transformations applied.
            pd.DataFrame: The DataFrame with scaled numeric columns.
        """
        # Ensure the base folder exists for saving outputs
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # 1. Identify numeric columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        scaling_summary = pd.DataFrame(columns=["Column", "Technique", "Original Min", "Original Max", "Original Mean"])

        # Choose the scaler
        if scaling_technique == "standard":
            scaler = StandardScaler()
        elif scaling_technique == "minmax":
            scaler = MinMaxScaler()
        elif scaling_technique == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling technique. Choose 'standard', 'minmax', or 'robust'.")
        
        # 2. Apply scaling to numeric columns
        df_scaled = df.copy()
        for col in numeric_cols:
            original_min = df[col].min()
            original_max = df[col].max()
            original_mean = df[col].mean()
            
            # Scale the column
            df_scaled[col] = scaler.fit_transform(df[[col]])
            
            # Append details to the summary
            scaling_summary = scaling_summary._append({
                "Column": col,
                "Technique": scaling_technique,
                "Original Min": original_min,
                "Original Max": original_max,
                "Original Mean": original_mean
            }, ignore_index=True)
        
        # 3. Save summary to a CSV
        summary_path = os.path.join(base_folder, "scaling_summary.csv")
        scaling_summary.to_csv(summary_path, index=False)
        print(f"Scaling summary saved to: {summary_path}")
        
        # 4. Generate a heatmap for scaled data
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_scaled[numeric_cols], cmap="coolwarm", cbar=True, xticklabels=numeric_cols)
        plt.title(f"Heatmap of Scaled Data ({scaling_technique.capitalize()})", fontsize=16)
        heatmap_path = os.path.join(base_folder, f"scaled_data_heatmap_{scaling_technique}.png")
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        print(f"Scaled data heatmap saved to: {heatmap_path}")
        
        return scaling_summary, df_scaled
    
    @staticmethod
    def apply_custom_encoding(df, encoding_dict, base_folder="encoding_analysis", ordinal_categories=None):
        """
        Encodes categorical columns in the DataFrame using specified encoding techniques for each column.

        Parameters:
            df (pd.DataFrame): The input DataFrame to encode.
            encoding_dict (dict): A dictionary mapping column names to encoding techniques.
                                Example: {"col1": "onehot", "col2": "ordinal", "col3": "label"}
            base_folder (str): Directory to save the summary and visualization of encoding.
            ordinal_categories (dict or None): Categories for ordinal encoding as a dictionary.
                                            Keys are column names, values are lists of categories in order.

        Returns:
            pd.DataFrame: A summary of the encoding transformations applied.
            pd.DataFrame: The DataFrame with encoded columns.
        """
        # Ensure the base folder exists for saving outputs
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # Initialize summary DataFrame
        encoding_summary = pd.DataFrame(columns=["Column", "Technique", "Unique Values"])
        df_encoded = df.copy()

        for col, technique in encoding_dict.items():
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in the DataFrame. Skipping.")
                continue

            unique_values = df[col].unique()

            if technique == "onehot":
                # OneHotEncoding
                encoder = OneHotEncoder(sparse=False, drop="first")  # Drop first to avoid multicollinearity
                encoded_df = pd.DataFrame(encoder.fit_transform(df[[col]]),
                                        columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]],
                                        index=df.index)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)

            elif technique == "ordinal":
                # OrdinalEncoding
                if ordinal_categories and col in ordinal_categories:
                    encoder = OrdinalEncoder(categories=[ordinal_categories[col]])
                else:
                    encoder = OrdinalEncoder()
                df_encoded[col] = encoder.fit_transform(df[[col]])

            elif technique == "label":
                # LabelEncoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col])

            else:
                print(f"Invalid encoding technique '{technique}' for column '{col}'. Skipping.")
                continue

            # Append details to the summary
            encoding_summary = encoding_summary._append({
                "Column": col,
                "Technique": technique,
                "Unique Values": unique_values
            }, ignore_index=True)

        # Save summary to a CSV
        summary_path = os.path.join(base_folder, "custom_encoding_summary.csv")
        encoding_summary.to_csv(summary_path, index=False)
        print(f"Encoding summary saved to: {summary_path}")
        
        # Heatmap of the encoded data
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_encoded.select_dtypes(include=["float64", "int64"]), cmap="coolwarm", cbar=True)
        plt.title(f"Heatmap of Encoded Data (Custom Encoding)", fontsize=16)
        heatmap_path = os.path.join(base_folder, "encoded_data_heatmap_custom.png")
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        print(f"Encoded data heatmap saved to: {heatmap_path}")
        
        return encoding_summary, df_encoded

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
# Short 
# Outliers / standrization / onehot encoding / pipeline / split / model 1 / metrics /save /hyper / evaluation /API /Fast API/ Docker


# Long           
# Def analytics for each variable -skweens / curtosis / numeric variables       
# Def input missing value column by column / each column with technic
# Corrleations analysis.
# ACP analysis / AFC /AFCM .....
# def outliers
# Api
# Docker

   
        
    

use = Eda('Outcome',"mean","mode" )
df = use.read_file().head(10)
print(df)
#use.data_diagnostic(df)
#use.univariate_analysis(df)
#use.multyvarie_analysis(df)
#use.input_missing_values(df)
# outlier_replacer = OutlierReplaceWithMedian(cols=[0, 2, 4, 5, 6, 7, 8], threshold=1.5)
# transformed_data = outlier_replacer.transform(df)
# transformed_df = pd.DataFrame(transformed_data, columns=df.columns)
# print(transformed_df)
# use.apply_scaling(df, base_folder="scaling_analysis", scaling_technique="standard")
encoding_dict = {"Pregnancies": "ordinal"}
use.apply_custom_encoding(df, encoding_dict, base_folder="encoding_analysis", ordinal_categories=None)
#use.target_variable_balance_check(df,'Outcome' )
#use.multyvarie_analysis(df,'Outcome')
# sns.catplot(x='Outcome'  , kind= 'count', data= df)
# plt.show()
