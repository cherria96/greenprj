# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:37:29 2022

@author: sujin
"""

import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from scipy import stats


path = 'C:/Users/cherr/OneDrive - postech.ac.kr/연구/VFA prediction/data/AD mappingv2.xlsx'
data = pd.read_excel(path, sheet_name = 0, index_col = 'Sample #')

#data preprocessing 
data = data.dropna()
data = data[data['COD'] >= data['sCOD']]
data = data[data['TKN'] >= data['TAN']]
data = data[data.columns[3:]]

VFA = ['HAc', 'HPr', 'HBu', 'HVa', 'HCa']

#data visualization
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (12,7))
axes = axes.flatten()
for n, item in enumerate(['HAc', 'HPr', 'HBu', 'HVa', 'HCa']):
    sns.histplot(data = data, x = item,ax = axes[n],)
fig.suptitle('Histogram of individual VFAs', y= 0.92)
plt.show()


#feature and target
targets = data[VFA]
targets = targets.replace(0, 0.0000001)
features = data.drop(targets, axis = 1)

#boxcox transformation
n=4
target = targets.iloc[:,n]
target_bc, lambd_HPr = stats.boxcox(target)
fig, axes = plt.subplots(1,2, figsize = (10,5))
sns.histplot(target, ax = axes[0])
sns.histplot(target_bc, ax = axes[1])
plt.suptitle('Box-Cox transformation of [{}]'.format(target.name))

#correlation
corr_matrix = data.corr()
cm = corr_matrix[target.name]
print(cm.sort_values(ascending=False))
mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, 
        square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



from model import reg, build_model

from sklearn.preprocessing import StandardScaler
standardscaler= StandardScaler()
X_train = standardscaler.fit_transform(features)
X_train = pd.DataFrame(X_train, columns = features.columns)
y_train = target_bc

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost as xgb
forest_reg = RandomForestRegressor()
xgb_reg = xgb.XGBRFRegressor()
ada_reg = AdaBoostRegressor()
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lin_reg = LinearRegression()
ridge_reg = Ridge()
lasso_reg = Lasso()
from sklearn.svm import SVR, NuSVR
sv_reg = SVR()
nusv_reg = NuSVR()

        

result = pd.DataFrame()
for n in range(0,len(targets.columns)):
    target = targets.iloc[:,n]
    target_bc, lambd_HPr = stats.boxcox(target)
    for j, y_train in enumerate([target, target_bc]):
        models = [forest_reg, xgb_reg, ada_reg, sv_reg, lin_reg, ridge_reg, lasso_reg ]
        scores = []
        for i in range(7):
            model = models[i]
            ml = reg(model, X_train, y_train)
            ml.model_fit(X_train, y_train, early_stopping=(False))
            compare, pred = ml.predict(X_train, y_train)
            score = ml.cross_val(X_train, y_train)
            model.score(X_train, y_train)
            scores.append(score)
        result['{0}-{1}'.format(j,target.name)] = scores
        
        
    
importance = ml.permutation_importance_plot(X_train, y_train)
