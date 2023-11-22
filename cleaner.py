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
    
    # @abstractmethod
    # def __merge_dataframe(self) -> pd.DataFrame:
    #     pass
        
    
    def read_csv(self, filename) -> pd.DataFrame:
        df = pd.read_csv(self.path_to_files + filename)
        return df

    def read_xls(self, filename) -> pd.DataFrame:
        xlsx = pd.ExcelFile(f"{self.path_to_files}{filename}")
        df_xls = pd.read_excel(xlsx)
        return df_xls
        
        
class FAOStatCleaner(CleanerABC):
    
    def __init__(self):
        self.dict_filename = {
            'TLU': 'FAOSTAT_agricultural_area.csv',
            'POC': 'FAOSTAT_production.csv'
        }
        self.list_dataframe_to_merge = []
    def __merge_dataframe(self) -> pd.DataFrame:
        return pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
    
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
        self.list_dataframe_to_merge.append(df_extract_TLU)
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
        df_csv['POC'] = df_csv['Value']
        del df_csv['Value']
        df_extract_POC = df_csv
        self.list_dataframe_to_merge.append(df_extract_POC)

        return df_extract_POC

    def cleanup(self):
        df_extract_TLU = self.__extract_TLU()
        df_extract_POC = self.__extract_POC()
        
        return self.__merge_dataframe()
        
class IMFStatCleaner(CleanerABC):
    
    def __init__(self):
        self.dict_filename = {
            'DEBT': 'IMF_debt.xls'
            }
        self.list_dataframe_to_merge = []
        
        
    def __extract_DEBT(self):
        
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
        
        # We need to convert all the index into numbers or NaN so that we can filter. "Special case: "Private debt, loans and debt securities (Percent of GDP)"
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

    def __merge_dataframe(self):
        pass
    
    def cleanup(self):
        return self.__extract_DEBT()
        
    
class UNStatCleaner(CleanerABC):
    
    def __init__(self):
        
        self.dict_filename = {
            'POP':'UN_population.csv'
        }
        
        self.list_dataframe_to_merge = []
        
    def __merge_dataframe(self) -> pd.DataFrame:
        pass
    
    def __extract_POPUL(self):
        
        filename = self.dict_filename['POP']
        df_csv = self.read_csv(filename)
        
        df_csv = df_csv.loc[df_csv['Year(s)'].between(1992,2021), ['Year(s)', 'Value']]

        df_csv.rename(columns={'Year(s)': 'Year', 'Value': 'POP'}, inplace=True)
        
        df_extract_POPUL = df_csv        
        return df_extract_POPUL

    def cleanup(self):
        return self.__extract_POPUL()

class WORLDBANKGROUPCleaner(CleanerABC):
    
    def __init__(self):
        
        self.dict_filename = {
            'RAIN': 'WORLDBANKGROUP_rain.csv', 
            'TEMP':'WORLDBANKGROUP_temperature.csv'
            }
        
        self.list_dataframe_to_merge = []
        
    def __merge_dataframe(self) -> pd.DataFrame:
        return pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
    
    def __extract_RAIN(self):
        
        filename = self.dict_filename['RAIN']
        df_csv = self.read_csv(filename)
        df_csv.drop('5-yr smooth', axis=1, inplace=True)
        df_csv.rename(columns={'Category': 'Year', 'Annual Mean': 'RAIN'}, inplace=True)
        df_extract_RAIN = df_csv.loc[(df_csv['Year']>= 1992 ) & (df_csv['Year']<=2021 )]

        return df_extract_RAIN
        
    def __extract_TEMP(self):
        
        filename = self.dict_filename['TEMP']
        df_csv = self.read_csv(filename)
        
        df_csv.drop('5-yr smooth', axis=1, inplace=True)
        df_csv.rename(columns={'Category': 'Year', 'Annual Mean': 'TEMP'}, inplace=True)
        df_extract_TEMP = df_csv.loc[(df_csv['Year']>= 1992 ) & (df_csv['Year']<=2021 )]

        return df_extract_TEMP
    
    def cleanup(self):
        self.list_dataframe_to_merge.append(self.__extract_RAIN())
        self.list_dataframe_to_merge.append(self.__extract_TEMP())
        
        return self.__merge_dataframe()
    
    
class MACROTRENDCleaner(CleanerABC):
    
    def __init__(self):
        
        self.dict_filename = {
            'ECO': 'MACROTREND_economic_growth.xls'
            }
    
        self.list_dataframe_to_merge = []
        
    def __extract_ECO(self)-> pd.DataFrame:
        filename = self.dict_filename['ECO']
        df_xls = self.read_xls(filename)
        
        df_xls.rename(columns={'Economic growth (%)': 'ECO'}, inplace=True)
        df_extract_ECO = df_xls

        return df_extract_ECO
            
    def __merge_dataframe(self) -> pd.DataFrame:
        pass

    def cleanup(self):
        return self.__extract_ECO()
        
    
class CleanerConcrete():
    
    def __init__(self):
        self.FAOStatCleaner = FAOStatCleaner()
        self.IMFStatCleaner = IMFStatCleaner()
        self.UNStatCleaner = UNStatCleaner()
        self.WORLDBANKGROUPCleaner = WORLDBANKGROUPCleaner()
        self.MACROTRENDCleaner = MACROTRENDCleaner()
        self.list_dataframe_to_merge = []
        
        
    def __merge_dataframe(self) -> pd.DataFrame:
        
        merged_df = pd.merge(self.list_dataframe_to_merge[0], self.list_dataframe_to_merge[1], on='Year')
        
        for index in range(2,len(self.list_dataframe_to_merge)):
            merged_df = pd.merge(merged_df, self.list_dataframe_to_merge[index], on='Year')
        
        return merged_df
           
    
    def cleanup(self):
        self.list_dataframe_to_merge.append(self.FAOStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.IMFStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.UNStatCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.WORLDBANKGROUPCleaner.cleanup())
        self.list_dataframe_to_merge.append(self.MACROTRENDCleaner.cleanup())
        return self.__merge_dataframe()
    
    def run(self):
        df_merged = self.cleanup()
        df_merged.drop('Year', axis=1, inplace=True)
        df_merged.to_csv('2d_DATA.csv', index=False)


if __name__ == '__main__':
    CleanerConcrete().run()
    
    pass