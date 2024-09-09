# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:57:41 2021

@author: sujin
"""
import pandas as pd
import matplotlib.pyplot as plt

usecols = 'E,I,L,M,S,T,W,Y,Z,AC,AD,AM,AN,AO,BC,BD'

def data_prep(filename, sheet_name, usecols= usecols):
    '''
    

    Returns data_prep in list 
    -------
    name_list : TYPE list
        data preprocessing 
        renaming colnames
        drop unused col, rows
        drop duplicates
    '''

    data = pd.read_excel(filename, sheet_name = sheet_name, usecols= usecols,
                         header = 2, index_col = 0, 
                         na_values= ['',' - ','NAN',0] )
    name_list = []
    for i in sheet_name:
        data[i].drop(data[i].index[range(0,5)], inplace = True)
        data[i].rename(columns={'DIG A': 'PS_vol', 'TS(%).1': 'PS_TS(%)', 'VS(%).1':'PS_VS(%)',
                                'Unnamed: 22':'Water ratio', 'Unnamed: 24':'Grit ratio', 'DIG A.1':'FWW_vol', 
                                'TS(%).2':'FWW_TS(%)', 'VS(%).2':'FWW_VS(%)', 'TS(%).3':'Dig_TS(%)','VS(%).3':'Dig_VS(%)',
                                'Nm3':'Biogas'}, inplace = True)
        data[i].dropna(inplace=True)
        data[i].drop_duplicates(['PS_TS(%)'], inplace=True)
        name_list.append(data[i])
    return (name_list)

def sheet_name():
    '''
    

    Returns sheet name in list
    -------
    sheet_name : TYPE list
        only limited to SBK data excel file

    '''
    sheet_name = []
    while True:
        i = input('which year? (type only the last digit) : ')
        if i == '':
            break
        sheet_name.append('Process Daily 201' + i)   
    return (sheet_name)


def comparison (y_pred, y_true, title):
    '''
    

    Parameters
    ----------
    y_pred : TYPE dataframe
        predicted y value
    y_true : TYPE dataframe
        true y value
    title : string
    ylabel : string, optional
        DESCRIPTION. The default is 'Biogas'.

    Returns
    -------
    None. (shows plot)

    '''
    comparison = pd.DataFrame(y_pred, columns= ['prediction_value'] )
    comparison['actual_value'] = pd.DataFrame(y_true)
    comparison['error'] = (abs(comparison['prediction_value']-comparison['actual_value']))/comparison['actual_value'] * 100
    fig, ax1 = plt.subplots(figsize = (12,8))
    ax1.plot(comparison.index, comparison.prediction_value, 'b')
    ax1.scatter(comparison.index, comparison.actual_value, color = 'red')
    ax1.legend(['prediction', 'actual'], loc= 2)
    ax2 = ax1.twinx()
    ax2.bar(comparison.index, comparison.error, color = 'skyblue', alpha = 0.5 )
    ax2.set_ylabel('error')
    ax2.set_ylim([0,200])
    ax2.legend(['error'])
    plt.title(title)
    plt.show()
    return comparison

def plot_history(history):
    

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.xlabel("epochs",)
    plt.ylabel("loss (%)", )
    plt.gca().set_ylim(0,100)
    plt.show()

def evaluate(model, x_test, y_test):
    import numpy as np
    predictions = model.predict(x_test)
    errors = abs(predictions - y_test)
    m = errors/ y_test
    m = m.replace([np.inf, np.nan], 0)
    mape = 100 * np.mean(m)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error : {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy 

def feature_importance_plot(model, features):
    import matplotlib.pyplot as plt
    import numpy as np
    n_features = features.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), features.columns )
    plt.xlabel("Random forest feature importance")
    plt.ylabel('feature')
    plt.ylim(-1, n_features)
    plt.show()
    
def permutation_importance_plot(model, X, y, features):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X, y, n_repeats= 20, random_state = 20, n_jobs = -1)
    forest_importances = pd.Series(result.importances_mean, index = features.columns)
    forest_importances.sort_values(inplace = True)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr = result.importances_std, ax = ax)
    ax.set_title("feature importances using permutation on model")
    ax.set_ylabel('Mean accuracy decrease')
    fig.tight_layout()
    plt.show()
    return forest_importances
        
    
    