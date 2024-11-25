# Class / method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pathlib 
import os
import random

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
df = use.read_file()
#use.data_diagnostic(df)
#use.univariate_analysis(df)
#use.multyvarie_analysis(df)
use.input_missing_values(df)
#use.target_variable_balance_check(df,'Outcome' )
#use.multyvarie_analysis(df,'Outcome')
# sns.catplot(x='Outcome'  , kind= 'count', data= df)
# plt.show()
