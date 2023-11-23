import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import math
'''
UPDATE ON VARIATION 3 

We will remove outliers data which is any data of 1973, 2002, 2003, 2004
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
    YEAR_FROM
        We will recover data that is starts from YEAR_FROM
    YEAR_END
        We will recover data that ends at YEAR_END
    YEAR_TO_REMOVE
        We will NOT recover data from this list of years because
            1. Incomplete dataset from FDI
            2. We will write this code for future improvements such as removal of outliers which we will do in variation 3
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
    YEAR_FROM = 1970
    YEAR_END = 2021
    YEARS_TO_REMOVE_OUTLIERS = [1973, 2002, 2003, 2004]
    YEAR_TO_REMOVE=[1991, 1992] + YEARS_TO_REMOVE_OUTLIERS
    
    @abstractmethod
    def cleanup() -> pd.DataFrame: 
        pass
    
    @abstractmethod
    def merge_dataframe(self) -> pd.DataFrame:
        pass        
    
    def read_csv(self, filename, skip_row=0) -> pd.DataFrame:
        if skip_row:
            '''
            Contex: For this project, we only use this when reading FDI file, so i'll just customize here 
                        
            Issue: There is some header or meta data that I can't see that is not allowing pandas to read the file
            
            Solution: We remove header and set the column manually
            '''
            column_name  = [
                "Country Name", "Country Code", "Indicator Name", "Indicator Code",
                "1960", "1961", "1962", "1963", "1964", "1965", "1966", "1967",
                "1968", "1969", "1970", "1971", "1972", "1973", "1974", "1975",
                "1976", "1977", "1978", "1979", "1980", "1981", "1982", "1983",
                "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991",
                "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999",
                "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007",
                "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015",
                "2016", "2017", "2018", "2019", "2020", "2021", "2022","2023"
            ]

            df = pd.read_csv(self.PATH_TO_FILES + filename, skiprows=[skip_row], header=None, names=column_name)
        else:
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
            'POC': 'FAOSTAT_production.csv'
        }
        self.list_dataframe_to_merge = []
        
    def merge_dataframe(self) -> pd.DataFrame: 
        return self.list_dataframe_to_merge[0]
    
  
            
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
        df_csv = df_csv.loc[(df_csv['Element'] == 'Production') & (df_csv['Year']>= self.YEAR_FROM) & (df_csv['Year'] <= self.YEAR_END) & (~df_csv['Year'].isin(self.YEAR_TO_REMOVE)), ['Year','Value']]     
        df_csv['POC'] = df_csv['Value']
        del df_csv['Value']
        df_extract_POC = df_csv

        #transform
        #1 unit to 1,000,000 unit
        df_extract_POC['POC'] /= 100000
        for index, row in df_extract_POC.iterrows():
            df_extract_POC.at[index, 'POC'] = math.log(row['POC'], math.e)
        self.list_dataframe_to_merge.append(df_extract_POC)

        return df_extract_POC

    def cleanup(self):
        df_extract_POC = self.__extract_poc()
        
        return self.merge_dataframe()
        
   
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
        
        df_csv = df_csv.loc[(df_csv['Year(s)']>= self.YEAR_FROM) & (df_csv['Year(s)'] <= self.YEAR_END) & (~df_csv['Year(s)'].isin(self.YEAR_TO_REMOVE)), ['Year(s)', 'Value']]

        df_csv.rename(columns={'Year(s)': 'Year', 'Value': 'POP'}, inplace=True)
        
        df_extract_POP = df_csv        
        
        #transform
        # 1. Change population unites from 1 thousand to 1 million
        df_extract_POP['POP'] /=1000

        
        # 2.  Ln it
        for index, row in df_extract_POP.iterrows():
            df_extract_POP.at[index, 'POP'] = math.log(row['POP'], math.e)
        return df_extract_POP

    def cleanup(self):
        return self.__extract_pop()

class WORLDBANKGROUPCleaner(CleanerABC):
    '''
    Methods:
    merge_dataframe(self) -> pd.DataFrame:
        There are data from other files, we will merged the dataframed based on the Year column
        
    __extract_fdi(self) -> pd.DataFrame
        Extract fdi data from the csv
        
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
            'TEMP':'WORLDBANKGROUP_temperature.csv',
            'FDI': 'WORLDBANKGROUP_FDI.csv'
            }
        
        self.list_dataframe_to_merge = []
        
    def merge_dataframe(self) -> pd.DataFrame:
        
        df_merge = pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
        return pd.merge(df_merge, self.list_dataframe_to_merge[2], on='Year')
    
    def __extract_rain(self):
        
        filename = self.dict_filename['RAIN']
        df_csv = self.read_csv(filename)
        df_csv.drop('5-yr smooth', axis=1, inplace=True)
        df_csv.rename(columns={'Category': 'Year', 'Annual Mean': 'RAIN'}, inplace=True)
        df_extract_RAIN = df_csv.loc[(df_csv['Year']>= self.YEAR_FROM ) & (df_csv['Year']<=self.YEAR_END ) & (~df_csv['Year'].isin(self.YEAR_TO_REMOVE))]
        
        #Transform
        #1. ln data
        for index, row in df_extract_RAIN.iterrows():
            df_extract_RAIN.at[index, 'RAIN'] = math.log(row['RAIN'], math.e)
        
        return df_extract_RAIN
        
    def __extract_temp(self):
        
        filename = self.dict_filename['TEMP']
        df_csv = self.read_csv(filename)
        
        df_csv.drop('5-yr smooth', axis=1, inplace=True)
        df_csv.rename(columns={'Category': 'Year', 'Annual Mean': 'TEMP'}, inplace=True)
        df_extract_TEMP = df_csv.loc[(df_csv['Year']>= self.YEAR_FROM ) & (df_csv['Year']<=self.YEAR_END ) & (~df_csv['Year'].isin(self.YEAR_TO_REMOVE))]
        
        #transform data
        #1. ln TEMP
        for index, row in df_extract_TEMP.iterrows():
            df_extract_TEMP.at[index,'TEMP'] = math.log(row['TEMP'])
        return df_extract_TEMP
    
    def __extract_fdi(self):
        
        filename = self.dict_filename['FDI']
        df_extract_fdi = self.read_csv(filename, skip_row=4)
        
        columns_of_interest = [
            "1970", "1971", "1972", "1973", "1974", "1975",
            "1976", "1977", "1978", "1979", "1980", "1981", "1982", "1983",
            "1984", "1985", "1986", "1987", "1988", "1989", "1990",
            "1993", "1994", "1995", "1996", "1997", "1998", "1999",
            "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007",
            "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015",
            "2016", "2017", "2018", "2019", "2020", "2021"
        ]
        columns_of_interest = [item for item in columns_of_interest if int(item) not in self.YEAR_TO_REMOVE]
        df_extract_fdi_iran = df_extract_fdi.loc[df_extract_fdi['Country Name'] == 'Iran, Islamic Rep.', columns_of_interest].T
        df_extract_fdi_iran['Year'] = df_extract_fdi_iran.index        
        df_extract_fdi_iran['FDI'] = df_extract_fdi_iran.iloc[:,0]
        
        df_extract_fdi_iran.drop(df_extract_fdi_iran.columns[0], axis=1, inplace=True)        
        df_extract_fdi_iran['Year'] = df_extract_fdi_iran['Year'].astype(int)  
        return df_extract_fdi_iran
    
    def cleanup(self):
        self.list_dataframe_to_merge.append(self.__extract_rain())
        self.list_dataframe_to_merge.append(self.__extract_temp())
        self.list_dataframe_to_merge.append(self.__extract_fdi())
        
        return self.merge_dataframe()
    
           
    
class CleanEverything(CleanerABC):
    '''
    Context: Main Cleaner that will run all the cleaning functions, build an aggregated dataframe and then write into the CSV file
    
    Attributes
        self.FAOStatCleaner:FAOStatCleaner
        self.UNStatCleaner:UNStatCleaner
        self.WORLDBANKGROUPCleaner:WORLDBANKGROUPCleaner
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
        self.UNStatCleaner = UNStatCleaner()
        self.WORLDBANKGROUPCleaner = WORLDBANKGROUPCleaner()
        
        self.list_dataframe_to_merge = []
        
        
    def merge_dataframe(self) -> pd.DataFrame:
        
        df_merged = pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
        
        for index in range(2,len(self.list_dataframe_to_merge)):
            df_merged = pd.merge(df_merged, self.list_dataframe_to_merge[index], on='Year')
        
        df_merged.drop('Year', axis=1, inplace=True)
        
        #Had a lot of issues with numpy.matmul because of the column order
        column_order = ['POC',	'POP',	'TEMP', 'FDI']
        
        df_merged_reordered = df_merged[column_order]
        return df_merged_reordered
           
    
    def cleanup(self) -> pd.DataFrame:
        self.list_dataframe_to_merge.append(self.FAOStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.UNStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.WORLDBANKGROUPCleaner.cleanup())

        return self.merge_dataframe()
    
    def run(self):
        df_merged = self.cleanup()
        df_merged.to_csv('variation_3_2D_DATA.csv', index=False)


if __name__ == '__main__':
    CleanEverything().run()
    