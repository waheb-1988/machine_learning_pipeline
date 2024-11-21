# Class / method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pathlib 
import os


class Eda :
    def __init__(self,target):
        self.target = target
        pass
      
    @staticmethod    
    def read_file():
        dir_folder= pathlib.Path.cwd().parent.parent
        input_path = dir_folder/ "data" / "data" 
        file_name = "diabetes.csv"
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
    def multyvarie_analysis(self,df,target):
        sns.pairplot(df,hue=target,size=3)
        plt.show()
        
    # Def input missing value
    # def outliers
    # def analysis of PCA
        
    

use = Eda('Outcome' )
df = use.read_file()
use.data_diagnostic(df)
use.univariate_analysis(df)
#use.target_variable_balance_check(df,'Outcome' )
#use.multyvarie_analysis(df,'Outcome' )
# sns.catplot(x='Outcome'  , kind= 'count', data= df)
# plt.show()
