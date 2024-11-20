# Class / method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pathlib 
import os


class Eda :
    def __init__(self):
        
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
        print("The variables names {x} ".format(x=list(df.columns.values)))

        column_headers =list(df.columns.values)
        qualitative_columns = [col for col in column_headers if df[col].dtype=="object"]
        quantitative_columns = [col for col in column_headers if df[col].dtype in ['int64', 'float64']]

        print("The qualitative variables {x} ".format(x=qualitative_columns))
        print("The quantitative variables {x} ".format(x=quantitative_columns))
        print("#"*50)


# use = Eda()
# df = use.read_file()
# use.data_diagnostic(df)