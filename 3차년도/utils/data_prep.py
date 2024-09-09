# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
# Load the Excel file

class DataCollection:
    def __init__(self, file_path, sheet_name = 0):
        
        super().__init__()
        xls = pd.ExcelFile(file_path)

        # Display the sheet names to identify the first sheet
        sheet_names = xls.sheet_names

        # Load the first sheet into a DataFrame
        self.df_first_sheet = pd.read_excel(file_path, sheet_name=sheet_names[sheet_name])

    def _data_prep(self, data):
        # draft.iloc[0].fillna(method='ffill', inplace= True)
        # col_names = [f'{col_2}_{col_3}' for col_2, col_3 in zip(draft.iloc[0], draft.iloc[1])]
        # data = draft[4:]
        # data.columns = col_names
        dates = pd.to_datetime(self.df_first_sheet["Date"], format='%Y-%m-%d')
        #dates = dates.dropna()
        data.index = dates
        data = data.apply(pd.to_numeric, errors='coerce', downcast='integer')
        data= data.loc[(data!=0).any(axis = 1)]
        # data= data.loc[:'2018-01-27']
        return data
    
    def build_data(self, data_type):
        '''
        Arg
        ---
        data_type: str ("substrate", "reactor", "sludge", "biogas")
        
        Output
        --- 
        DataFrame
        '''
        if data_type == "substrate":
            '''
            col_names = ["input_ton", "screw_pH", "screw_TS", "screw_VS/TS", 
            "press1_pH", "press1_TS", "press1_VS/TS", "press2_pH", "press2_TS", 
            "press2_VS/TS", "FeCl2"]    
            '''
            substrate_draft = self.df_first_sheet.iloc[:, 1:12]
            data = self._data_prep(substrate_draft)
        elif data_type == "reactor":
            '''
            col_names = ["volume", "HRT", "recircle", 
             "Rpress_vol", "Rpress_TS", "Rpress_retake", "Rpress_out", 
             "temp_in", "temp_out"]
            '''
            reactor_draft = self.df_first_sheet.iloc[:, 12:22]
            data = self._data_prep(reactor_draft)
        elif data_type == "sludge":
            '''
            col_names = ["temp", "pH", "TS", "VS/TS", "VFA", 
            "B-Alkalinity", "Alkalinity", "TCOD", "sCOD", "NH3-N"]
            '''
            sludge_draft = self.df_first_sheet.iloc[:, 22:34]
            sludge_data = self._data_prep(sludge_draft)
            data = sludge_data.iloc[:, [0,1,2,3,4,7,8,9,10,11]]
        elif data_type == "biogas":
            '''
            col_names = ["production", "CH4", "H2S", "CO2", "O2"]
            '''
            biogas_draft = self.df_first_sheet.iloc[:, 34:39]
            biogas_data = self._data_prep(biogas_draft)
            data = biogas_data.iloc[:, [0,1,2,3,4]]
        else:
            print("Invalid data type")
        return data

# interpolation 
class Interpolation:
    def __init__(self, data):
        self.data = data
        
    def monointerpolation(self, df):
        monointerpolate = interpolate.PchipInterpolator(df.index, df)
        interpolate_df = monointerpolate(self.data.index)
        return interpolate_df
    
    def visualize(self, df, interpolate_df):
        plt.plot(df.index, df, 'ko', 
                 self.data.index, interpolate_df, 'r--')
        plt.show()
        
    def get_output(self, col, interpolation_type = "mono"):
        df = self.data[col].dropna()
        if interpolation_type == "mono":
            interpolate_df = self.monointerpolation(df)
        else:
            print("Invalid interpolation type ")
        self.visualize(df, interpolate_df)
        return interpolate_df
    


