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
        
    @abstractmethod
    def read_file(name:str):
        pass
    
    @abstractmethod
    def cleanup() -> pd.DataFrame: 
        pass
        
        
class FAOStatCleaner(CleanerABC):
    
    def __init__(self):
        self.list_filename = ['FAOSTAT_agricultural_area.csv', 'FAOSTAT_employment_in_agriculture.csv', 'FAOSTAT_production.csv']
    
    #write brief description of what TLU means in the equation
    def __extract_TLU(self) -> pd.DataFrame:
        
        pass
    
    def __extract_POC(self) -> pd.DataFrame:
        
        pass
            
    
    def cleanup(self):
        pass

class IMFStatCleaner(CleanerABC):
    
    def __init__(self):
        self.list_filename = ['IMF_debt.xls', 'IMF_unemployment.xls']
        
    def __extract_DEBT(self):
        
        pass

    
    def cleanup(self):
        
    
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
    pass


if __name__ == '__main__':
    pass