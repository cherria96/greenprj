# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:45:28 2024

@author: Sujin
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from statsmodels.tsa.seasonal import STL
import seaborn as sns
from statsmodels.graphics import tsaplots

# Helper function for time series analysis
# Data overview
class DataOverview:
    """
    This class overviews basic data statistics and visualize datasets.

    Attributes:
    - data: Dataframe
    - data_type: str
        Either one of them ["substrate", "reactor", "sludge", "biogas"]
    
    Methods:
    - stats(): void
        print basic statistics of dataset 
    - draw_plot(columns: List or any iterable objects, 
                drop_na: Bool, figszie: tuple,
                **kwargs: optional for scatter plot)
        draw basic scatter plot of datasets
    """
    def __init__(self, data, data_type):
        self.data = data
        self.data_type = data_type
    
    def stats(self):
        data_stats = self.data.describe()
        print(f"\nstatistics ({self.data_type}):")
        print(data_stats)
    
    def draw_plot(self, columns, drop_na= False, figsize = (12,8), **kwargs):
        fig, axs = plt.subplots(nrows = len(columns), ncols = 1, 
                                figsize = figsize)
        for i, column in enumerate(columns):
            if drop_na:
                df = self.data.dropna(subset = column)
            else: df = self.data.copy()
            axs[i].scatter(df.index, df[column], **kwargs)
            axs[i].set_title(f'{self.data_type}-{column} time series data')
            axs[i].set_xlabel('date')
            axs[i].set_ylabel('value')
        plt.tight_layout()
        plt.show()




# anaomaly detection
class AnomalyDetection:
    """
    Detect anomaly data points in dataset 

    Attributes:
    - data: Dataframe 
    - target_col: str
    - outliers_fraction: float
    - figsize: tuple

    Methods:
    - CART(): return model
    - visualize(result, anomaly_type): 
    """
    def __init__(self, data, target_col, outliers_fraction = float(0.03), figsize = (12,8)):
        self.data = data
        self.target_col = target_col
        self.figsize = figsize
        
        scaler = StandardScaler()
        _scaled = scaler.fit_transform(self.data[target_col].values.reshape(-1,1))
        self.scaled_df = pd.DataFrame(_scaled)
        
        self.outliers_fraction = outliers_fraction
    
    def CART(self, ):
        model = IsolationForest(contamination=self.outliers_fraction)
        model.fit(self.scaled_df.dropna())
        return model
        
    def STL(self):
        
        pass

    def ClusteringAD(self):
        pass

    def visualize(self, result, anomaly_type):
        fig, axs = plt.subplots(nrows = len(self.target_col), ncols = 1, figsize = self.figsize)
        for i, column in enumerate(self.target_col):
            anomaly = result.loc[result['anomaly'] == -1, [column]]
            axs[i].plot_date(result.index, result, color = 'black', linestyle = '--', label = 'Normal')
            axs[i].plot(anomaly.index, anomaly[column], 'ro', label = 'Anomaly')
        axs[-1].legend() 
        plt.suptitle(f'Outlier detection with {anomaly_type}')
        plt.show()
        return anomaly

    def get_output(self, anomaly_type):
        if anomaly_type == "CART":
            model = self.CART()
        elif anomaly_type == "STL":
            model = self.STL()
        elif anomaly_type == "Cluster":
            model = self.ClusteringAD()
        else:
            print("Invalid anomaly detection type")
        result = self.data.copy()
        print("prediction result", model.predict(self.scaled_df.dropna()))
        result['anomaly'] = model.predict(self.scaled_df.dropna())
        anomaly_data = self.visualize(result, anomaly_type)
        return anomaly_data

        

# time series analysis
class TimeseriesAnalysis:
    """
    Attribute
    - data: Dataframe 
    - period: int
        time series analysis period 
    
    Method
    - get_STLplot(col: str): void
    - get_seasonalplot()

    """
    def __init__(self, data, period = 7):
        self.data = data
        self.period = period
        
    def _STLdecomposition(self, col):
        decomposition = sm.tsa.seasonal_decompose(self.data[col].dropna(), 
                                                       model='additive',
                                                       period=self.period)
        return decomposition
    
    def get_STLplot(self, col):
        decomposition = self._STLdecomposition(col)
        decomposition.plot()
        plt.show()
    
    def get_seasonal_corr(self, columns, **kwargs):
        seasonal_dict = {
            col: self._STLdecomposition(col).seasonal for col in columns 
        }
        seasonality_corr = pd.DataFrame(seasonal_dict).corr()
        sns.clustermap(seasonality_corr, **kwargs)
        plt.show()
    
    def get_trend_corr(self, columns, **kwargs):
        trend_dict = {
            col: self._STLdecomposition(col).trend for col in columns 
        }
        trend_corr = pd.DataFrame(trend_dict).corr()
        sns.clustermap(trend_corr, **kwargs)
        plt.show()
    
    def get_autocorrelation_plot(self, col, lags = 70):
        '''
        when a clear trend exists in a time series,
        the autocorrelation tends to be high at small lags
        when seasonality exists, 
        the autocorrelation goes up periodically at larger lags

        If the height of the bars is outside the region,
        the correlation is statistically significant
        '''
        tsaplots.plot_acf(self.data[col].dropna(), lags = lags)
        plt.title(f"Autocorrelation of {col}")
        plt.xlabel("Lag at k")
        plt.ylabel("Correlation Coefficient")
        plt.show()
        
    def get_partial_plot(self, col, lags = 70):
        tsaplots.plot_pacf(self.data[col].dropna(), lags = lags)
        plt.title(f"Partial Autocorrelation of {col}")
        plt.xlabel("Lag at k")
        plt.ylabel("Correlation Coefficient")
        plt.show()


    

    
    

        
'''

class EDA:
    def __init__(self, data, name, col):
        self.datacollection = DataCollection
        self.name = name
        self.col = col
        
    def stats(self):
        data_stats = self.data.describe()
        print(f"\nstatistics ({self.name}):")
        print(data_stats)
    
    def vis(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.data.index, self.data[self.col], label=self.name)
        plt.title(f'{self.name} time series data')
        plt.xlabel('date')
        plt.ylabel('value')
        plt.legend()
        plt.show()
    
    def trend_analysis(self):
        # 추세 분석 (Lowess Smoothing)
        lowess = sm.nonparametric.lowess
        data_smoothed = lowess(self.data[self.col], self.data.index, frac=0.1)
        original_length = len(self.data)
        smoothed_length = len(data_smoothed)
        original_index = self.data.index
        smoothed_index = original_index[:smoothed_length]
        
        plt.figure(figsize=(15, 6))
        plt.plot(self.data.index, self.data[self.col], label=self.name)
        plt.plot(smoothed_index, data_smoothed[:, 1], label='Lowess Smoothing')
        plt.title(f'{self.name} Trend analysis')
        plt.xlabel('date')
        plt.ylabel('value')
        plt.legend()
        plt.show()
    
    def _anomaly_detection(self, df, result):
        plt.rc('figure',figsize=(12,8))
        plt.rc('font',size=15)
        print(type(result))
        fig = result.plot()

        fig, (ax1,ax2) = plt.subplots(2, sharex = True)
        argmax = np.argmax(df)
        argmin = np.argmin(df)
        x = result.resid.index
        y = result.resid.values
        ax1.plot_date(x, df, color = 'black', linestyle = '--')
        ax1.plot(x[argmax], df[argmax], 'ro', markersize = 15, alpha = 0.5)
        ax1.plot(x[argmin], df[argmin], 'ro', markersize = 15, alpha = 0.5)
        ax1.set_title('{} value'.format(self.col))
        ax2.plot_date(x, y, color = 'black')
        ax2.annotate('Anomaly', (mdates.date2num(x[argmax]), y[argmax]), xytext=(30, 20), 
                   textcoords='offset points', color='red',arrowprops=dict(facecolor='red',arrowstyle='fancy'))
        ax2.axhline(2, color = 'red', linestyle = '--', alpha = 0.4)
        ax2.axhline(0, color = 'black', linestyle = '--', alpha = 0.4)
        ax2.axhline(-2, color = 'red', linestyle = '--', alpha = 0.4)
        ax2.set_title('Residue result of {} value'.format(self.col))
        fig.tight_layout()
        plt.show()
    
    def cart_anomaly_detection(self, df):
        outliers_fraction = float(0.1)
        scaler = StandardScaler()
        _scaled = scaler.fit_transform(df.values.reshape(-1,1))
        data = pd.DataFrame(_scaled)
        model = IsolationForest(contamination=outliers_fraction)
        model.fit(data)
        df['anomaly'] = model.predict(data)
        fig, ax = plt.subplots(figsize = (12,8))
        anomaly = df.loc[df['anomaly'] == -1, [self.col]]
        ax.plot_date(df.index, df, color = 'black', linestyle = '--', label = 'Normal')
        ax.plot(anomaly.index, anomaly[self.col], 'ro', label = 'Anomaly')
        plt.legend()
        plt.title('Outlier detection with CART')
        plt.show()

    def seasonal_decomp(self):
        decomposition = sm.tsa.seasonal_decompose(self.data[self.col].dropna(), model='additive', period=365)
        # self._anomaly_detection(self.data[[self.col]].dropna(), decomposition)
        self.cart_anomaly_detection(self.data[[self.col]].dropna())
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(15, 10))

        plt.subplot(4, 1, 1)
        plt.plot(self.data.index, self.data[self.col], label=self.name)
        plt.title(f'{self.name} data')

        plt.subplot(4, 1, 2)
        plt.plot(trend, label='Trend')
        plt.title('Trend')

        plt.subplot(4, 1, 3)
        plt.plot( seasonal, label='Seasonal')
        plt.title('Seasonal')

        plt.subplot(4, 1, 4)
        plt.plot(residual, label='Residual')
        plt.title('Residual')

        plt.tight_layout()
        plt.show()
    

#%%
# 시계열 분석 수행
substrate_EDA = EDA(substrate_data, 'substrate', col= 'screw_pH')
substrate_EDA.seasonal_decomp()
#%%



reactor_EDA = EDA(reactor_data, 'reactor', col='sludge_ton')
reactor_EDA.seasonal_decomp()
#%%



sludge_EDA = EDA(sludge_data, 'sludge', col = 1)
sludge_EDA.seasonal_decomp()





biogas_EDA = EDA(biogas_data, 'biogas')
biogas_EDA.seasonal_decomp()




# %%
'''