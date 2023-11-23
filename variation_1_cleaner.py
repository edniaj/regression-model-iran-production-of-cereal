import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import math
'''
This file serves the purpose of extracting raw files from the raw_data folder, aggregate the data and build a CSV file for the purpose of machine learning
'''
class CleanerABC(ABC):
    """
    Abstract base class for data cleaning operations
    
    Context:
    Due to the problem statement (agriculture in Iran), there are limited datasets scattered around different online databases / reports.
    We have created an ABC to standardised the cleaning operations from files with different structures and formats due to the database
    
    Attributes:
    PATH_TO_FILES (str)
        This is path relative to this directory
    
    Abstract Methods:
    cleanup() -> pd.DataFrame
        Main method to clean and extract raw data, then build and return a dataframe
    
    Methods:
    read_csv(filename: str) -> pd.DataFrame :
        Read a csv file given the filename with respect to the raw_data folder that is in the same directory as this file
        
    read_xls(filename: str) -> pd.DataFrame :
        Read a xls file given the filename with respect to the raw_data folder that is in the same directory as this file
        
    
    """
    PATH_TO_FILES = './raw_data/'
            
    @abstractmethod
    def cleanup() -> pd.DataFrame: 
        pass
    
    @abstractmethod
    def merge_dataframe(self) -> pd.DataFrame:
        pass        
    
    def read_csv(self, filename) -> pd.DataFrame:
        df = pd.read_csv(self.PATH_TO_FILES + filename)
        return df

    def read_xls(self, filename) -> pd.DataFrame:
        xlsx = pd.ExcelFile(f"{self.PATH_TO_FILES}{filename}")
        df_xls = pd.read_excel(xlsx)
        return df_xls
        
        
class FAOStatCleaner(CleanerABC):
    
    '''
    Methods:
    merge_dataframe(self) -> pd.DataFrame:
        There are data from other files, we will merged the dataframed based on the Year column
        
    __extract_tlu(self) -> pd.DataFrame:
        Extract Total Land Used in aggriculture data from the csv
    
    __extract_poc(self) -> pd.DataFrame:
        Extract Production of Cereal data from the csv
    
    cleanup(self) -> pd.DataFrame:
        Main method to clean and extract raw data, then build and return a dataframe
    '''    
    def __init__(self):
        self.dict_filename = {
            'TLU': 'FAOSTAT_agricultural_area.csv',
            'POC': 'FAOSTAT_production.csv'
        }
        self.list_dataframe_to_merge = []
        
    def merge_dataframe(self) -> pd.DataFrame: 
        return pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
    
    def __extract_tlu(self) -> pd.DataFrame:
        
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
        #transform
        #1. convert from 1unit to 1000 

        df_extract_TLU['TLU'] /= 1000
        self.list_dataframe_to_merge.append(df_extract_TLU)
        return df_extract_TLU       
            
    def __extract_poc(self) -> pd.DataFrame:
        
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
        df_csv['POC'] = df_csv['Value']
        del df_csv['Value']
        df_extract_POC = df_csv

        #transform
        #1 unit to 1,000,000 unit
        df_csv['POC'] /= 1000000
        self.list_dataframe_to_merge.append(df_extract_POC)

        return df_extract_POC

    def cleanup(self):
        df_extract_TLU = self.__extract_tlu()
        df_extract_POC = self.__extract_poc()
        
        return self.merge_dataframe()
        
class IMFStatCleaner(CleanerABC):
    
    '''
    Methods:
    merge_dataframe(self) -> pd.DataFrame:
        There are data from other files, we will merged the dataframed based on the Year column
        
    __extract_debt(self) -> pd.DataFrame:
        Extract Private Debt, loans and securities as a percent of GDP from the csv
    
    cleanup(self) -> pd.DataFrame:
        Main method to clean and extract raw data, then build and return a dataframe    

    Additional Info - Data arrangement for this xls file is peculiar. Take note
    '''
    
    def __init__(self):
        self.dict_filename = {
            'DEBT': 'IMF_debt.xls'
            }
        self.list_dataframe_to_merge = []
        
        
    def __extract_debt(self):
        
        '''
        This xls file structure is weird.
        It looks like this 
        
        Private debt 1950 1951 1952 ... 2021 2022
        <BLANK>
        Iran  no data no data .... some data 
        
        We need to work around it
        '''
        
        filename = self.dict_filename['DEBT']
        df_xls = self.read_xls(filename)
        
        # We need to transpose it because of weird file structure
        df_xls = df_xls.T
        
        # We need to convert all the index into numbers or NaN so that we can filter. 
        # "Special case: "Private debt, loans and debt securities (Percent of GDP)"
        df_xls.index = pd.to_numeric(df_xls.index, errors='coerce')
        df_xls = df_xls.loc[(df_xls.index >= 1992) & (df_xls.index <= 2021) ]
        
        list_index = df_xls.index
        
        convert_to_df = {
            'Year': [],
            'DEBT': []
        }
        for i in list_index:
            convert_to_df['DEBT'].append(df_xls[1][i])
            convert_to_df['Year'].append(int(i))
            
        
        df_extract_DEBT = pd.DataFrame(convert_to_df)
        
        return df_extract_DEBT

    def merge_dataframe(self):
        pass
    
    def cleanup(self):
        return self.__extract_debt()
        
    
class UNStatCleaner(CleanerABC):
    
    '''
    Methods:
    merge_dataframe(self) -> pd.DataFrame:
        There are data from other files, we will merged the dataframed based on the Year column
        
    __extract_pop(self) -> pd.DataFrame:
        Extract Population of Iran from the csv
    
    cleanup(self) -> pd.DataFrame:
        Main method to clean and extract raw data, then build and return a dataframe    
    '''
    
    def __init__(self):
        
        self.dict_filename = {
            'POP':'UN_population.csv'
        }
        
        self.list_dataframe_to_merge = []
        
    def merge_dataframe(self) -> pd.DataFrame:
        pass
    
    def __extract_pop(self):
        
        filename = self.dict_filename['POP']
        df_csv = self.read_csv(filename)
        
        df_csv = df_csv.loc[df_csv['Year(s)'].between(1992,2021), ['Year(s)', 'Value']]

        df_csv.rename(columns={'Year(s)': 'Year', 'Value': 'POP'}, inplace=True)
        
        df_extract_POP = df_csv        
        
        #transform
        # 1. Change population unites from 1 thousand to 1 million
        df_extract_POP['POP'] /= 1000
        # 2. Squaring it
        df_extract_POP['POP'] *= df_extract_POP['POP']

        return df_extract_POP

    def cleanup(self):
        return self.__extract_pop()

class WORLDBANKGROUPCleaner(CleanerABC):
    '''
    Methods:
    merge_dataframe(self) -> pd.DataFrame:
        There are data from other files, we will merged the dataframed based on the Year column
        
    __extract_rain(self) -> pd.DataFrame:
        Extract rain data from the csv

    __extract_temp(self) -> pd.DataFrame:
        Extract temp data from the csv
    
    cleanup(self) -> pd.DataFrame:
        Main method to clean and extract raw data, then build and return a dataframe  
    '''
    def __init__(self):
        
        self.dict_filename = {
            'RAIN': 'WORLDBANKGROUP_rain.csv', 
            'TEMP':'WORLDBANKGROUP_temperature.csv'
            }
        
        self.list_dataframe_to_merge = []
        
    def merge_dataframe(self) -> pd.DataFrame:
        
        return pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
    
    def __extract_rain(self):
        
        filename = self.dict_filename['RAIN']
        df_csv = self.read_csv(filename)
        df_csv.drop('5-yr smooth', axis=1, inplace=True)
        df_csv.rename(columns={'Category': 'Year', 'Annual Mean': 'RAIN'}, inplace=True)
        df_extract_RAIN = df_csv.loc[(df_csv['Year']>= 1992 ) & (df_csv['Year']<=2021 )]
        
        #Transform
        #1. ln data
        for index, row in df_extract_RAIN.iterrows():
            df_extract_RAIN.at[index, 'RAIN'] = math.log(row['RAIN'])
        
        return df_extract_RAIN
        
    def __extract_temp(self):
        
        filename = self.dict_filename['TEMP']
        df_csv = self.read_csv(filename)
        
        df_csv.drop('5-yr smooth', axis=1, inplace=True)
        df_csv.rename(columns={'Category': 'Year', 'Annual Mean': 'TEMP'}, inplace=True)
        df_extract_TEMP = df_csv.loc[(df_csv['Year']>= 1992 ) & (df_csv['Year']<=2021 )]
        
        #transform data
        #1. ln TEMP
        for index, row in df_extract_TEMP.iterrows():
            df_extract_TEMP.at[index,'TEMP'] = math.log(row['TEMP'])
        return df_extract_TEMP
    
    def cleanup(self):
        self.list_dataframe_to_merge.append(self.__extract_rain())
        self.list_dataframe_to_merge.append(self.__extract_temp())
        
        return self.merge_dataframe()
    
    
class MACROTRENDCleaner(CleanerABC):
    '''
    Methods:
    merge_dataframe(self) -> pd.DataFrame:
        There are data from other files, we will merged the dataframed based on the Year column
        
    __extract_pop(self) -> pd.DataFrame:
        Extract Population of Iran from the csv
    
    cleanup(self) -> pd.DataFrame:
        Main method to clean and extract raw data, then build and return a dataframe  
    '''  
    def __init__(self):
        
        self.dict_filename = {
            'ECO': 'MACROTREND_economic_growth.xls'
            }
    
        self.list_dataframe_to_merge = []
        
    def __extract_eco(self)-> pd.DataFrame:
        filename = self.dict_filename['ECO']
        df_xls = self.read_xls(filename)
        
        df_xls.rename(columns={'Economic growth (%)': 'ECO'}, inplace=True)
        df_extract_ECO = df_xls
        
        #transform data
        #1. absolute(ECO)
        df_extract_ECO['ECO'] = round(np.abs(df_extract_ECO['ECO']), 6)
        
        return df_extract_ECO
            
    def merge_dataframe(self) -> pd.DataFrame:
        pass

    def cleanup(self):
        return self.__extract_eco()
        
    
class CleanEverything(CleanerABC):
    '''
    Context: Main Cleaner that will run all the cleaning functions, build an aggregated dataframe and then write into the CSV file
    
    Attributes
        self.FAOStatCleaner:FAOStatCleaner
        self.IMFStatCleaner:IMFStatCleaner
        self.UNStatCleaner:UNStatCleaner
        self.WORLDBANKGROUPCleaner:WORLDBANKGROUPCleaner
        self.MACROTRENDCleaner:MACROTRENDCleaner
        self.list_dataframe_to_merge:List[pd.DataFrame]
    
    methods:
        merge_dataframe()-> pd.DataFrame:
            Merge all the dataframes based on the column 'Year' but we will drop this column after merging since it's not needed in the machine learning model
        cleanup()-> pd.DataFrame
            Clean up all the raw data and then recover all the built dataframes, and then we merge
        run():
            Run clean up, recover dataframe and then write into csv file
    
    '''
    def __init__(self):
        self.FAOStatCleaner = FAOStatCleaner()
        self.IMFStatCleaner = IMFStatCleaner()
        self.UNStatCleaner = UNStatCleaner()
        self.WORLDBANKGROUPCleaner = WORLDBANKGROUPCleaner()
        self.MACROTRENDCleaner = MACROTRENDCleaner()
        self.list_dataframe_to_merge = []
        
        
    def merge_dataframe(self) -> pd.DataFrame:
        
        df_merged = pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
        
        for index in range(2,len(self.list_dataframe_to_merge)):
            df_merged = pd.merge(df_merged, self.list_dataframe_to_merge[index], on='Year')
        
        df_merged.drop('Year', axis=1, inplace=True)
        
        #Had a lot of issues with numpy.matmul because of the column order
        column_order = ['POC', 'TEMP', 'TLU', 'RAIN', 'POP', 'DEBT', 'ECO']
        
        df_merged_reordered = df_merged[column_order]
        return df_merged_reordered
           
    
    def cleanup(self) -> pd.DataFrame:
        self.list_dataframe_to_merge.append(self.FAOStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.IMFStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.UNStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.WORLDBANKGROUPCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.MACROTRENDCleaner.cleanup())
        return self.merge_dataframe()
    
    def run(self):
        df_merged = self.cleanup()
        df_merged.to_csv('variation_1_2D_DATA.csv', index=False)


if __name__ == '__main__':
    CleanEverything().run()
    