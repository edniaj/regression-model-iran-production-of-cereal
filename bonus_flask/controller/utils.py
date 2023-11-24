import pandas as pd

PATH_TO_CSV = './public/'

def max_r2_score_row(df: pd.DataFrame) -> dict:
    max_index = df['R2_SCORE'].idxmax()    
    df_max_value = df.loc[df.index == max_index]
    result_dict = df_max_value.to_dict('records')[0]

    return result_dict
    

def read_csv(filename) -> dict: 
    
    df = pd.read_csv(f'{PATH_TO_CSV}{filename}')
    return max_r2_score_row(df)

