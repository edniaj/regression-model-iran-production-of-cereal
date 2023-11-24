import pandas as pd


'''
This file serves to only read the csv file
'''


'''

PATH_TO_CSV
    Relative path to the csv file (this file has all the r2 values ). Do not change inside the code.
max_r2_score_row(df: pd.DataFrame) -> dict
    Only takes the highest R2_Score and convert it to dict, easier to work with

read_csv(filename:str) -> dict:
    Reads the file based on relative path set by PATH_TO_CSV
'''

PATH_TO_CSV = './public/'

def max_r2_score_row(df: pd.DataFrame) -> dict:
    max_index = df['R2_SCORE'].idxmax()    
    df_max_value = df.loc[df.index == max_index]
    result_dict = df_max_value.to_dict('records')[0]

    return result_dict
    

def read_csv(filename) -> dict: 
    
    df = pd.read_csv(f'{PATH_TO_CSV}{filename}')
    return max_r2_score_row(df)

