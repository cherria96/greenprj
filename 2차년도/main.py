# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:46:18 2023

@author: Sujin
"""
#-*-coding:utf-8 -*-
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.font_manager as fm
from matplotlib import font_manager
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

# font_fname = 'C:/Users/Sujin/AppData/Local/Microsoft/Windows/Fonts/KoPubWorld Dotum Medium.ttf'
# font_family = font_manager.FontProperties(fname=font_fname).get_name()
# plt.rc('font', family=font_family)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# csv_path= 'D:/OneDrive - postech.ac.kr/연구/Transfer learning/data/Dat_Fullscale(230517).csv'
csv_path = '/Users/sujinchoi/Library/CloudStorage/OneDrive-postech.ac.kr/그린 뉴딜/코드/Full_data(2020~2023).csv'
df_raw = pd.read_csv(csv_path)
date_time = pd.to_datetime(df_raw.pop('일자'), format='%Y.%m.%d')
df_raw = df_raw.drop(['가축분 투입량','돈모 투입량', '동물성잔재 투입량','반건식 소화조 온도', '반건식 소화조 pH', '반건식 소화조 EC', '반건식 소화조 DO (mg/L)',
       '반건식 소화조 TS (%)', '반건식 소화조 VS (%)', '반건식 소화조 ORP (mV)'], axis = 1)

def plot_features(df,cols,time):
  plot_features = df[cols]
  plot_features.index = time
  _ = plot_features.plot(style = 'o', subplots=True, ms = 3, title = cols, legend = False, figsize = (10,10))
  plt.tight_layout()
  plt.show()
##data preprocessing##
summary_raw = df_raw.describe().transpose()

#0이 오류값인 성상 --> na 값으로 대체 --> na 값 3개 이하 항목 ffillna 로 결측치 보완
lst = ['온도', 'pH', 'EC', 'DO', 'TS', 'VS', 'ORP', 'CH4', 'CO2', 'O2']
for l in lst:
  for col in df_raw.columns:
    if l in col:
      df_raw[col] = df_raw[col].replace(0, np.nan)
print('0 value\n', df_raw.isin([0]).sum())

for col in df_raw.columns:
    if '음식물 탈리액' in col:
        df_raw[col] = df_raw[col].fillna(method = 'ffill')

gas = ['CH4', 'CO2', 'O2']
for g in gas:
    for col in df_raw.columns:
        if g in col:
            df_raw[col] = df_raw[col].fillna(method = 'ffill')
            
#ORP: 양수 제거 
plot_cols = []
for col in df_raw.columns:
  if 'ORP' in col:
    plot_cols.append(col)
    print(df_raw[col][np.abs(df_raw[col]) == df_raw[col]])
plot_features(df_raw, plot_cols, date_time)

#pH 이상치 제거
plot_cols = []
for col in df_raw.columns:
  if 'pH' in col:
    plot_cols.append(col)
    print(df_raw[col][df_raw[col]>9]) #음식물 탈리액: 17.5, 습식소화조2: 80.9

FW_pH = df_raw['음식물 탈리액 pH']
bad_pH = FW_pH == 17.5
FW_pH[bad_pH] = 7.50

AD2_pH = df_raw['습식소화조2 pH']
bad_pH = AD2_pH == 80.9
AD2_pH[bad_pH] = 8.09

bad_pH = AD2_pH == 3.33
AD2_pH[bad_pH] = 8.33

plot_df = df_raw[plot_cols]
plot_df.index = date_time
_ = plot_df.plot(style = 'o', subplots=True, ms = 3, title = plot_cols, legend = False, figsize = (10,10))
plt.tight_layout()

#O2 이상치 제거
AD1_O2 = df_raw['습식 소화조1 O2 (%)']
bad_O2 = AD1_O2 == '1..9'
AD1_O2[bad_O2] = 1.9

df_raw['습식 소화조1 O2 (%)'] = df_raw['습식 소화조1 O2 (%)'].astype('float')
#H2S 이상치 제거
plot_cols = []
for col in df_raw.columns:
  if 'H2S' in col:
    plot_cols.append(col)
    df_raw[col] = df_raw[col].astype(np.float)
    df_raw[col]=df_raw[col].replace(0, np.nan)
    
AD2_H2S = df_raw['습식 소화조2 H2S (ppm)']
bad_H2S = AD2_H2S == 26464
AD2_H2S[bad_H2S] = 2646

plot_H2S = df_raw[plot_cols]
plot_H2S.index = date_time
_ = plot_H2S.plot(style = 'o', subplots=True, ms = 3, title = plot_cols, legend = False, figsize = (10,10))
plt.tight_layout()

#가수분해조 투입량 이벤트 구간 na 처리
df_raw['가수분해조 투입량'] = df_raw['가수분해조 투입량'].replace(0, np.nan)

summary = df_raw.describe().transpose()

##data visualization

#투입량
plot_cols = ['가축분뇨 투입량', '음식물 탈리액 투입량', '혼합액저장조 투입량', '가수분해조 투입량', '습식소화조1 투입량', '습식소화조2 투입량']
plot_features(df_raw,plot_cols,date_time)

#주중 투입량
weekday_time = date_time[(date_time.dt.dayofweek!=5) & (date_time.dt.dayofweek!=6)]
plot_features(df_raw.iloc[weekday_time.index], plot_cols, weekday_time)

#습식소화조2 투입량 이상치 제거
AD2_in = df_raw['습식소화조2 투입량']
bad_in = AD2_in == 627
AD2_in[bad_in] = 62

#온도
plot_cols = []
for col in df_raw.columns:
  if '온도' in col:
    plot_cols.append(col)
plot_features(df_raw, plot_cols, date_time)

#바이오가스 총 생산량 = 바이오가스 사용량 + 잉여가스 사용량
df_raw['바이오가스생산량'] = df_raw['바이오가스 사용량'] + df_raw['잉여가스 사용량']
df_raw['바이오가스생산량'] = df_raw['바이오가스생산량'] * 0.86                    #바이오가스 총 생산량 86%가 습식소화조로부터 생산되었다고 가정
biogas = df_raw['바이오가스생산량']
biogas.index = date_time                                                        
biogas.plot(style = 'o', ms = 3,legend = False, figsize = (10,5))
plt.title('바이오가스 생산량')
plt.show()

#메탄 생산량
plot_cols = []
for col in df_raw.columns:
  if 'CH4' in col:
    plot_cols.append(col)
plot_features(df_raw, plot_cols, date_time)


#투입량 물질 수지
def sum_weekly(df, cols):                                                      #주간 투입량 계산
    sub_in = pd.DataFrame()
    for col in cols:
        _in = []
        sum_col = 0
        for i in df.index:
            if i % 7 == 0:
                _in.append(sum_col)
                sum_col = df[col].iloc[i]
            else:
                sum_col += df[col].iloc[i]
        sub_in[str(col)] = _in[1:]
    return sub_in

cols = []
for col in df_raw.columns:
  if '투입량' in col:
    cols.append(col)
sub_in = sum_weekly(df_raw, cols)
weekly = date_time[date_time.dt.dayofweek==0]
sub_in.index = weekly
plot_features(sub_in, cols, weekly)
    
sub_in['투입량 차이'] =  sub_in['가축분뇨 투입량'] - sub_in['음식물 탈리액 투입량'] - sub_in['혼합액저장조 투입량']
sub_in['투입량 차이'].plot(style = 'o', ms = 3,legend = False, figsize = (10,5))
plt.title('주간 투입량 물질수지 (가축분뇨 - 음식물 탈리액- 혼합액저장조 투입량)')
plt.show()

sub_in['투입량 차이'] =  sub_in['혼합액저장조 투입량'] - sub_in['가수분해조 투입량'] - sub_in['반건식 소화조 투입량']
sub_in['투입량 차이'].plot(style = 'o', ms = 3,legend = False, figsize = (10,5))
plt.ylim([-2000,2000])
plt.title('주간 투입량 물질수지 (혼합액저장조 - 가수분해조 - 반건식 소화조 투입량)')
plt.show()

sub_in['투입량 차이'] =  sub_in['가수분해조 투입량'] - sub_in['습식소화조1 투입량'] - sub_in['습식소화조2 투입량']
sub_in['투입량 차이'].plot(style = 'o', ms = 3,legend = False, figsize = (10,5))
plt.ylim([-200,200])
plt.title('주간 투입량 물질수지 (가수분해조 - 습식소화조)')
plt.show()



df_final = df_raw.drop(['바이오가스 사용량', '잉여가스 사용량'], axis = 1)
df_final.index = date_time

##BIOGAS PREDICTION##
#1) Univariate time series prediction : ARIMA model-----------------------------------------------------------------------#
#correlation
corr_mat = df_final.corr()
biogas_corr = corr_mat['바이오가스생산량'][abs(corr_mat['바이오가스생산량'])>0.3].sort_values()
biogas_corr[:-1].plot.bar()
plt.title('바이오가스생산량 간 상관관계 (p < 0.05)')
plt.show()

corr_p = pd.DataFrame(index = ['corr', 'p_value'])
for col in biogas_corr.index[:-1]:
    if '혼합액' in col:
        continue
    elif '가수분해조 투입량' in col:
        continue
    corr = stats.pearsonr(df_final['바이오가스생산량'],df_final[col])
    corr_p[str(col)] = [corr[0], corr[1]]   #모든 항목 p value < 0.05 --> 상관성 유의미
    
#Check stationary of biogas time series data
#1) Plotting Rolling Statistics : We find rolling mean and variance to check stationary
#2) Dickey-Fuller Test: if test statistic < critical value : time series is stationary

ts = df_final['바이오가스생산량']
# adfuller library 
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
    return result

# check_mean_std
def check_mean_std(ts):
    #Rolling statistics
    rolmean = ts.rolling(window = 7).mean()
    rolstd = ts.rolling(window = 7).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("시간")
    plt.ylabel("바이오가스생산량")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()

check_mean_std(ts) #not constant mean, std
check_adfuller(ts) #test statistic > critical value
#We can conclude that the data is not stationary

#Change data to stationary data
#1) Moving average method
window_size = 7
moving_avg = ts.rolling(window_size).mean()
plt.figure(figsize=(22,10))
plt.plot(ts, color = "red",label = "Original")
plt.plot(moving_avg, color='black', label = "moving_avg_mean")
plt.title("바이오가스생산량")
plt.xlabel("시간")
plt.ylabel("바이오가스생산량")
plt.legend()
plt.show()

ts_moving_avg_diff = ts - moving_avg
ts_moving_avg_diff.dropna(inplace=True) # first 7 is nan value due to window size
# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_moving_avg_diff)
check_adfuller(ts_moving_avg_diff) #test statistic < critical values in 1% => stationary series with 99% confidence

#2) Differencing method
ts_diff = ts - ts.shift()
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("시간")
plt.ylabel("바이오가스생산량 차분")
plt.show()

ts_diff.dropna(inplace=True) # due to shifting there is nan values
# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_diff)
check_adfuller(ts_diff) #test statistic < critical values in 1% => stationary series with 99% confidence


#Time series prediction 
#ts_diff chosen for time series prediction
# ACF and PACF 
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=10)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

# ACF
#q = 1, cross upper confidence interval for the first time
plt.figure(figsize=(22,10))
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

# PACF
#p = 1, cross upper confidence interval for the first time
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Therefore, we use (1,0,1) as parameters of ARIMA models and predict

# ARIMA LİBRARY
from statsmodels.tsa.arima.model import ARIMA
#중간에 빈 시간(2020-07 ~ 2020-11) 기점으로 데이터 분리
ts_post =ts['2020-12-01':]
# fit model
model = ARIMA(ts, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit = model.fit()

# predict
forecast = model_fit.predict(start='2021-07-01', end='2021-12-31')

# visualization
plt.figure(figsize=(22,10))
plt.plot(ts.index, ts,label = "original")
plt.plot(forecast,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("시간")
plt.ylabel("바이오가스 생산량")
plt.legend()
plt.show()

# predict all path
from sklearn.metrics import mean_squared_error
import math
# fit model
model2 = ARIMA(ts, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit2 = model2.fit()
forecast2 = model_fit2.predict()
error = mean_squared_error(ts, forecast2)
print("error (RMSE): " ,math.sqrt(error))
# visualization
plt.figure(figsize=(22,10))
plt.plot(ts.index, ts,label = "original")
plt.plot(forecast2,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("시간")
plt.ylabel("바이오가스 생산량")
plt.legend()
plt.savefig('graph.png')
plt.show()


steps = 31
fcast = model_fit2.get_forecast(steps = steps)
fcast_sum = fcast.summary_frame(alpha = 0.10)
forecast_idx = pd.date_range('2022-01-01', periods = steps)
fcast_sum.index = forecast_idx

fig, ax = plt.subplots(figsize=(15, 5))
# Plot the data (here we are subsetting it to get a better look at the forecasts)
ts['2021-10-01':].plot(ax=ax)
# Construct the forecasts
fcast_sum['mean'].plot(ax=ax, style='k--')
ax.fill_between(fcast_sum.index, fcast_sum['mean_ci_lower'], fcast_sum['mean_ci_upper'], color='k', alpha=0.1);
plt.title('2022년 1월 바이오가스 생산량 예측 (신뢰도 90%)')
plt.show()

#2) Multivariate time series prediction : VAR, XGBOOST-----------------------------------------------------------------------#

#2-1)VAR
#corr value > 0.3인 항목을 input variable 
input_var = biogas_corr.index.to_list()
input_var.remove('혼합액저장조 온도')
input_var.remove('가수분해조 투입량')  #Missing data 없는 항목 대상

nobs = 6
df = df_final[input_var]
train_df = df[:-nobs]
test_df = df[-nobs:]

#Granger's Causality test
#H0: the past values of time series(X) do not cause the other series(Y)
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=30
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

#Granger's Casulaity test assumes that data is stationary      
def transform_stationary(data):
    data_diff = data - data.shift()
    plt.figure(figsize=(22,10))
    plt.plot(data_diff)
    plt.title("Differencing method") 
    plt.xlabel("시간")
    plt.ylabel(data.name + "차분")
    plt.show()
    
    data_diff.dropna(inplace=True) # due to shifting there is nan values
    return data_diff

df_VAR = pd.DataFrame()
for col in input_var:
    print('\n')
    ts = df_final[col]
    while True: 
        print(str(col), 'stationary check')
        result = check_adfuller(ts)
        if (result[0] < result[4]['1%']) and (result[1] < 0.05):
            df_VAR[col] = ts
            break
        else:
            ts = transform_stationary(ts)

#모두 1차 differencing 함
df_VAR = pd.DataFrame()
for col in input_var:
    df_VAR[col] = transform_stationary(train_df[col])

        
GC_matrix = grangers_causation_matrix(df_VAR, variables = df_VAR.columns)        
sns.heatmap(GC_matrix, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
plt.title("Granger's Causality 결과")
plt.show()
df_biogas = GC_matrix.iloc[-1]
df_biogas = df_biogas[df_biogas <0.049] #음식물 탈리액 pH, 습식소화조1 CH4, 가축분뇨 투입량, 습식소화조1,2 투입량
lst = df_biogas.index.to_list()
idx = []
for i in lst:
    idx.append(i.split('_')[0])
idx.append('바이오가스생산량')

df_VAR = df_VAR[idx]


#Cointegration Test: establish the presence of statistically significant connection between two or more time series
#Order of integration: the number of differencing required to make a non-stationary time series stationary

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df_VAR) #모두 공적분 관계임을 확인함


from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(df_VAR)

#Lag order (p = 6)
aic = []
bic = []
fpe = []
hqic = []
for i in range(15):
    result = model.fit(i)
    aic.append(result.aic)
    bic.append(result.bic)
    fpe.append(result.fpe)
    hqic.append(result.hqic)
    
ax1= plt.subplot(2,2,1)
plt.plot(range(15), aic)
plt.title('AIC')

ax2 = plt.subplot(2,2,2)
plt.plot(range(15), bic, label = 'bic')
plt.title('BIC')


plt.subplot(2,2,3, sharex = ax1)
plt.plot(range(15), fpe, label = 'fpe')    
plt.title('FPE')
plt.xlabel('Lag order')

plt.subplot(2,2,4, sharex = ax2)
plt.plot(range(15), hqic, label = 'hqic')
plt.title('HQIC')
plt.xlabel('Lag order')
plt.show()

model_fitted = model.fit(6)
model_fitted.summary()

from statsmodels.stats.stattools import durbin_watson
def adjust(val, length= 6): return str(val).ljust(length)

out = durbin_watson(model_fitted.resid)
for col, val in zip(df_VAR.columns, out):
    print(adjust(col), ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_VAR.values[-lag_order:] #initial value for the forecast
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df[idx].index[-nobs:], columns=df_VAR.columns + '_1d')
df_forecast

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(train_df[idx], df_forecast, second_diff=False)        

fig, axes = plt.subplots(nrows=int(1), ncols=2, dpi=150, figsize=(10,5))
for i, (col,ax) in enumerate(zip(['습식 소화조1 CH4 (%)','바이오가스생산량' ], axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test_df[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})


print('Forecast Accuracy of: biogas production')
accuracy_prod = forecast_accuracy(df_results['바이오가스생산량_forecast'].values, test_df['바이오가스생산량'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('Forecast Accuracy of: methane ')
accuracy_prod = forecast_accuracy(df_results['습식 소화조1 CH4 (%)_forecast'].values, test_df['바이오가스생산량'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))
    
    
results = model_fitted
irf = results.irf(6)
irf.plot(orth=False, response = '바이오가스생산량')
irf.plot_cum_effects(orth=False, response = '바이오가스생산량')

irf.plot(orth=False, response = '습식 소화조1 CH4 (%)')

