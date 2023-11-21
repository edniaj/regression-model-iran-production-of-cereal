import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
'''
write documentation and ABC
'''


'''
Notes:
Need to delete some csv files if we are totally 100% not using the data and variables anymore
'''
class CleanerABC(ABC):
    
    path_to_files = './raw_data/'
            
    @abstractmethod
    def cleanup() -> pd.DataFrame: 
        pass
    
    def read_csv(filename) -> pd.DataFrame:
        df = pd.read_csv(self.path_to_files + filename)
        return df

    def read_xls (filename) -> pd.DataFrame:
        df = pd.read_xls(self.path_to_files + filename)
        return df
        
        
class FAOStatCleaner(CleanerABC):
    
    def __init__(self):
        self.dict_filename = {
            'TLU': 'FAOSTAT_agricultural_area.csv',
            'POC': 'FAOSTAT_production.csv'
        }
    
    
    #write brief description of what TLU means in the equation
    def __extract_TLU(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_to_files + self.dict_filename['TLU'])
        print(df)
        pass
    
    def __extract_POC(self) -> pd.DataFrame:
        
        pass
            
    
    def cleanup(self):
        self.__extract_TLU()
        pass

class IMFStatCleaner(CleanerABC):
    
    def __init__(self):
        self.list_filename = ['IMF_debt.xls', 'IMF_unemployment.xls']
        
    def __extract_DEBT(self):
        
        pass

    
    def cleanup(self):
        pass
        
    
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
    
    def run(self):
        self.FAOStatCleaner.cleanup()


if __name__ == '__main__':
    Cleaner().run()
    pass