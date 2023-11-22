import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
'''
write documentation and ABC
'''


'''
Notes:
Need to delete some csv files if we are totally 100% not using the data and variables anymore
Do we normalize in this file and transform in this file ?

'''
class CleanerABC(ABC):
    
    path_to_files = './raw_data/'
            
    @abstractmethod
    def cleanup() -> pd.DataFrame: 
        pass
    
    def read_csv(self, filename) -> pd.DataFrame:
        df = pd.read_csv(self.path_to_files + filename)
        return df

    def read_xls(self,filename) -> pd.DataFrame:
        # df = pd.pd.read_excel(self.path_to_files + filename)
        return df
        
        
class FAOStatCleaner(CleanerABC):
    
    def __init__(self):
        self.dict_filename = {
            'TLU': 'FAOSTAT_agricultural_area.csv',
            'POC': 'FAOSTAT_production.csv'
        }
    
    
    #write brief description of what TLU means in the equation
    def __extract_TLU(self) -> pd.DataFrame:
        
        '''
        1. Removal of unused data
        There are 3 types of land 
            1. Arable land
            2. Permanent crops
            3. Permanent meadows and pastures
                - Permanent meadows and pastures should be removed because it isn't used for production of ceral, it's meant for livestocks
        
        2. Feature manipulating
            We will create a new varaible called TLU (Total land used) which consist of Arable land and Permanent crops
        
        return column [Year, TLU]
        '''
        
        filename = self.dict_filename['TLU']
        df_csv = self.read_csv(filename)
        
        df_csv = df_csv.loc[df_csv['Item'] != 'Permanent meadows and pastures']
        df_csv = df_csv.loc[(df_csv['Year'] >= 1992) & (df_csv['Year'] <= 2021)]
                
        df_extract_TLU = df_csv.groupby('Year')['Value'].sum().reset_index(name='TLU')
        return df_extract_TLU       
        
    
    def __extract_POC(self) -> pd.DataFrame:
        
        '''
        
        1. Removal of unused data
        There are 3 types of Element 
            1. Area harvested
            2. Yield
            3. Production
                - We only want production
        
        return column [Year, Production]
        '''
        
        filename = self.dict_filename['POC']
        df_csv = self.read_csv(filename)

        df_csv = df_csv.loc[(df_csv['Element'] == 'Production') & (df_csv['Year']>= 1992) & (df_csv['Year'] <= 2021), ['Year','Value']]     
        df_csv['Production'] = df_csv['Value']
        del df_csv['Value']
        df_extract_POC = df_csv

        return df_extract_POC

    def cleanup(self):
        df_extract_TLU = self.__extract_TLU()
        df_extract_POC = self.__extract_POC()
        df_merged_TLU_POC = pd.merge(df_extract_TLU, df_extract_POC, on='Year')
        # print(df_merged_TLU_POC)
        return df_merged_TLU_POC
        
class IMFStatCleaner(CleanerABC):
    
    def __init__(self):
        self.dict_filename = {
            'DEBT': 'IMF_debt.xls'
            }
        
    def __extract_DEBT(self):
        filename = self.dict_filename['DEBT']
        df_xls = pd.read_xls(filename)
        print('test')
        print(df_xls)
        pass

    
    def cleanup(self):
        self.__extract_DEBT()
        
    
class UNStatCleaner(CleanerABC):
    
    def __init__(self):
        self.list_filename = ['UN_population.csv']
        
    def __extract_POPUL(self):
        
        pass

class WORLDBANKGROUPCleaner(CleanerABC):
    
    def __init__(self):
        self.list_filename = ['WORLDBANKGROUP_rain.csv', 'WORLDBANKGROUND_temperature.csv']
    
    def __extract_RAIN(self):
        pass
    
    def __extract_teemperature(self):
        pass
    
    
class MACROTRENDCleaner(CleanerABC):
    
    def __init__(self):
        self.list_filename = ['MACROTREND_economic_growth.csv']
    

class Cleaner:
    
    def __init__(self):
        self.FAOStatCleaner = FAOStatCleaner()
        self.IMFStatCleaner = IMFStatCleaner()
    def merge_dataframes():
        pass
    
    def run(self):
        self.FAOStatCleaner.cleanup()
        self.IMFStatCleaner.cleanup()


if __name__ == '__main__':
    Cleaner().run()
    pass