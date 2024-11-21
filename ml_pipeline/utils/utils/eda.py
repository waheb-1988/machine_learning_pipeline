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
    def univarie_analysis(df):
        # Barchart
        # Boxplot
        # densite
        
    
        pass
    
    def multyvarie_analysis(self,df,target):
        sns.pairplot(df,hue=target,size=3)
        plt.show()
        
    

use = Eda('Outcome' )
df = use.read_file()
use.data_diagnostic(df)
use.numeric_analysis(df)
#use.target_variable_balance_check(df,'Outcome' )
use.multyvarie_analysis(df,'Outcome' )
# sns.catplot(x='Outcome'  , kind= 'count', data= df)
# plt.show()
