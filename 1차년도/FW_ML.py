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
import joblib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import os

path = 'E:/OneDrive - postech.ac.kr/연구/VFA prediction/code/Finalv1.xlsx'
data = pd.read_excel(path, sheet_name = 0, index_col = 0)
data = data[data['Sample'].str.contains('M')]

#data preprocessing 
data = data[data['COD'] >= data['sCOD']]
data = data[data['TKN'] >= data['TAN']]
data = data[data.columns[3:]]
data['IA/PA'] = data['IA']/data['PA']

data = data.fillna(0)
data.reset_index(inplace = True, drop = True)

#feature and target
feature = []
for i in data.columns:
    if 'FW' in i:
        feature.append(i)
feature.extend(['TA', 'IA', 'PA', 'pH', 'IA/PA', ])
features = data[feature]
VFA = ['TVFA']
targets = data[VFA]
targets = targets.replace(0, 0.0000001)

data.reset_index(inplace = True, drop = True)
data = data.drop(np.where((data.F == 0) & (data.Cl == 0))[0])
targets = data[['HAc', 'HPr', 'HBu', 'HVa', 'HCa']]
targets = targets.replace(0, 0.0000001)
features = data.drop(targets, axis = 1)
features.reset_index(inplace = True, drop = True)

#boxcox transformation (figure 1)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(nrows =1, ncols = 2, figsize = (20,10))
axes = axes.flatten()
for n in range(0, len(targets.columns)*2, 2):
    target = targets.iloc[:,int(n/2)]
    target_bc, lambd_HPr = stats.boxcox(target)
    sn = sns.histplot(target, ax = axes[n],  kde = True, color = 'gray', bins = 10)
    sn.lines[0].set_color('crimson')
    axes[n].xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[n].yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[n].set(xlabel = '[{}_0] g/L'.format(target.name))

    sn = sns.histplot(target_bc, ax = axes[n+1], kde = True, color = 'gray', bins = 10)
    sn.lines[0].set_color('crimson')
    axes[n+1].set(xlabel = '[{}_1] g/L'.format(target.name))
    axes[n+1].set_yticks(np.arange(0,30,5))
plt.savefig('fig1.png', dpi = 600, bbox_inches = 'tight')


#correlation
plt.rcParams.update({'font.size': 20})

corr_matrix = train.corr()
cm = corr_matrix[target.name]
print(cm.sort_values(ascending=False))
mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(30,30))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, 
        square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


from functions_2 import split_data, comparison2, comparison, scaling, display_scores, plot_learning_curves
from functions import feature_importance_plot
from model import reg, build_model
from sklearn.preprocessing import StandardScaler
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
#forest_reg = RandomForestRegressor(n_estimators = 110 , min_samples_split= 8, min_samples_leaf= 2, max_features = 'auto', max_depth = 50, bootstrap = True,)
X_train, X_test, y_train, y_test, x_train, x_test, scaler = split_data(features, targets, 0.3)
x_train.reset_index(inplace = True, drop = True)

X_train.drop(np.where(x_train['FW_Protein'] <0)[0], inplace = True)
y_train.drop(np.where(x_train['FW_Protein'] <0)[0], inplace = True)
x_train.drop(np.where(x_train['FW_Protein'] <0)[0], inplace = True)
X_train.drop(7, inplace = True)
y_train.drop(7, inplace = True)
x_train.drop(7, inplace = True)

train = pd.concat([x_train, y_train], axis = 1)


#[Fig1]data visualization
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20,10))
axes = axes.flatten()
for n, (item, label) in enumerate(zip(['TVFA', 'TA', 'pH'], ['[TVFA] g/L', r'[TA] g/L $\mathregular{CaCO_3}$', 'pH'])):
    p = sns.histplot(data = train, x = item,ax = axes[n],)
    p.set_xlabel(label)
plt.savefig('fig2.png', dpi = 600, bbox_inches = 'tight')




def pred_result___(targets, features, models, save = False, folder = 'model'):
    result = pd.DataFrame()
    for n in range(0,len(targets.columns)):
        target = targets.iloc[:,n]
        target_np = target.to_numpy()
        target_bc, lambd = stats.boxcox(target_np)
        for j, y in enumerate([target_np, target_bc]):
            X_train, X_test, y_train, y_test = split_data(features, y, 0.2)
            scores = []
            for i in range(len(models)):
                model = models[i]
                if j == 1:
                    ml = reg(model, X_train, y_train, lambd)
                else:
                    ml = reg(model, X_train, y_train, None)
                ml.model_fit(X_train, y_train, early_stopping=(False))
                compare, pred = ml.predict(X_train, y_train)
                score = ml.cross_val(X_train, y_train)
                print(model.score(X_train, y_train))
                scores.append(score)
                if save == True:
                    filename = folder + '/{}_{}_{}.sav'.format(str(model).split('(')[0], target.name, j)
                    joblib.dump(model, filename)
            result['{0}-{1}'.format(j,target.name)] = scores
            importance = ml.permutation_importance_plot(X_train, y_train)
    return result, importance





def shap_val(model, features):
    import shap
    explainer = shap.TreeExplainer(model, )
    shap_val = explainer(features)
    shap_val_mean={}
    for i, name in zip(range(0, shap_val.values.shape[-1]), features.columns):
        shap_val_mean[name] = abs(shap_val.values[:,i]).mean()
    shap_val_mean = {k:v for k, v in sorted(shap_val_mean.items(), key = lambda item : item[1], reverse = True)}
    return shap_val_mean

def pred_result(target, features, model, save = False, folder = 'model'):
    target_np = target.to_numpy()
    target_bc, lambd = stats.boxcox(target_np)
    scores = []
    for j, y in enumerate([target_np,target_bc]):
        if j == 1:
            ml = reg(model, features, y, lambd)
        else:
            ml = reg(model, features, y, None)
        ml.model_fit(features, y, early_stopping=(False))
        compare, pred = ml.predict(features, y)
        score = ml.cross_val(features, y)
        print(model.score(features, y))
        scores.append(score)
        if save == True:
            if not os.path.isdir(folder):
                os.makedirs(folder)
            filename =  folder + '/{}_{}_{}.sav'.format(str(model).split('(')[0], target.name, j)
            joblib.dump(model, filename)
    importance = ml.permutation_importance_plot(features, y)
    return scores, importance, model, compare


plt.rcParams.update({'font.size': 10})
  
models = [forest_reg, xgb_reg, ada_reg, sv_reg, lin_reg, ridge_reg, lasso_reg]
n_targets = targets[['TVFA']]
n_targets_bc, lambd = stats.boxcox(n_targets.iloc[:,0])
n_features = features.drop(['TVFA + EtOH'], axis = 1)


result, importance = pred_result(targets, features, models)
sorted_features = importance.sort_values(ascending = False).index



# supp table 1,2
report = pd.DataFrame()
for target in y_train.columns:
    target = y_train.loc[:,target]
    scores = []
    for model in models:
        result, importance, _,compare = pred_result(target, X_train,model)
        sorted_features = importance.sort_values(ascending = False).index
        selected_feat = X_train.loc[:, sorted_features[0:10]]
        score, _, _,compare = pred_result(target, selected_feat, model)
        scores.append(score)
    report['{}-0'.format(target.name)] = np.array(scores)[:,0]
    report['{}-1'.format(target.name)] = np.array(scores)[:,1]

# feature selection figure 2
target = y_train['TVFA']
model_type = xgb_reg
result, importance, model, compare = pred_result(target, X_train[['IA/PA', 'FW_Hac',]], model_type)
#sorted_features = importance.sort_values(ascending = False).index
sorted_features = shap_val(model, X_train)
feature_select = {}
for i in range(1, len(sorted_features)+1):
    selected_feat = X_train.loc[:, sorted_features[0:i]]
    #selected_feat = X_train.loc[:, list(sorted_features.keys())[0:i]]
    result, _, model, compare = pred_result(target, selected_feat, model_type)
    feature_select[selected_feat.columns[-1]] = result
FS_df = pd.DataFrame.from_dict(feature_select).T
sns.set(style = "white", font_scale =2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size':22})
plt.figure(figsize = (20,12))
plt.title('Prediction score of [{}] when cumulating each feature'.format(target.name), pad = 20, fontdict = {'fontsize' : 25})
sns.lineplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 1, color = 'b', linewidth = 4)
sns.scatterplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 1, markers = 'o', s=300 )
sns.lineplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 0, color = 'orange', linewidth = 4)
sns.scatterplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 0, markers = 'o', s=300 ) 
plt.legend(['{}_1'.format(target.name), '{}_0'.format(target.name)],loc = 4 )
plt.ylabel('R squared', )
plt.xlabel('Added features')
plt.show()
final_feat = selected_feat.iloc[:, :2]
result, importance, model, compare = pred_result(target, X_train[['IA/PA', 'FW_TKN', 'FW_Hac']], xgb_reg, save = True, folder = 'model/final' )
target_np = target.to_numpy()
y_train_bc, lambd = stats.boxcox(target_np)
plot_learning_curves(model,final_feat, y_train_bc)
    
#[fig 3] final prediction of TVFA
X_test_final= X_test[['IA/PA','FW_Hac']]
Y_test = y_test['TVFA']
target_np = Y_test.to_numpy()
Y_test_bc, lambd = stats.boxcox(target_np)
Y_pred_bc = model.predict(X_test_final)
Y_pred = inv_boxcox(Y_pred_bc, lambd)
comparison2(Y_pred, Y_test, title = 'prediction with XGBRFRegressor (test data)')
compare_test = comparison(Y_pred_bc, Y_test, title = 'prediction with XGBRFRegressor (test data)', lambd = None)
print("final model r2 :", model.score(X_test_final, Y_test))

features_scaled = scaler.transform(features)
features_scaled = pd.DataFrame(features_scaled, columns = features.columns)
input_var = features_scaled[['IA/PA', 'FW_Hac']]
model_TVFA = joblib.load('E:/OneDrive - postech.ac.kr/연구/VFA prediction/model/final/XGBRFRegressor_TVFA_1.sav')
output_real = data.TVFA.to_numpy()
output_real_bc, lambd_output = stats.boxcox(output_real)
output_bc = model_TVFA.predict(input_var)
output = inv_boxcox(output_bc, lambd_output)
comparison2(output, output_real, title = 'prediction with XGBRFRegressor (whole dataset)')
compare_test = comparison(output, output_real, title = 'prediction with XGBRFRegressor (whole dataset)', lambd = None)
print("final model r2 :", model_TVFA.score(input_var, output_real_bc))

##[Fig5] SHAP value##
import shap

explainer = shap.TreeExplainer(model, )
shap_val = explainer( X_test[['IA/PA', 'FW_Hac']])
shap_val = explainer( input_var)
shap_val = explainer(X_train[['IA/PA', 'FW_Hac', 'FW_TKN']])

    #1) summary plot : mean shap value of each features
shap.summary_plot(shap_val,X_train, show = False)
plt.savefig('fig/shap_summary_raw.png', dpi = 600, bbox_inches = 'tight')

    #2) whether each feature impact positively or negatively on the model output
no = 31
shap.plots.waterfall(shap_val[no], show = False)
plt.savefig('fig/shap_waterfall{}.png'.format(str(no)), dpi = 600, bbox_inches = 'tight')

    #3) to see each individual feature on the general model output (SHAP determines the color automatically, that has the most irrelevant impact on the feature)
feature = 'FW_Hac'
shap.plots.scatter(shap_val[:, feature], color = shap_val, show = False) 
plt.savefig('fig/shap_Hac.png', dpi = 600, bbox_inches = 'tight')

    #*4) to see the positive/negative impact of degree of feature value on SHAP  (RECOMMENDED!)
shap.plots.beeswarm(shap_val)

shap_val.data[:,0] = x_train['IA/PA'].to_numpy()
shap_val.data[:,1] = x_train['FW_Hac'].to_numpy()

# acetate, propionate prediction
target = y_train['HPr']
X_train.drop('TVFA + EtOH', axis =1, inplace = True)
result, importance, model, compare = pred_result(target, X_train, xgb_reg)
sorted_features = shap_val(model, X_train)
feature_select = {}
for i in range(1, len(sorted_features)+1):
    model = xgb_reg
    #selected_feat = X_train.loc[:, sorted_features[0:i]]
    selected_feat = X_train.loc[:, list(sorted_features.keys())[0:i]]
    result, _, model, compare = pred_result(target, selected_feat, model)
    feature_select[selected_feat.columns[-1]] = result
FS_df = pd.DataFrame.from_dict(feature_select).T
sns.set(style = "white", font_scale =2)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size':30})
plt.figure(figsize = (30,15))
plt.title('Prediction score of [{}] when adding each feature'.format(target.name), pad = 20, fontdict = {'fontsize' : 30})
sns.lineplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 1, color = 'b', linewidth = 4)
sns.scatterplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 1, markers = 'o', s=300 )
sns.lineplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 0, color = 'orange', linewidth = 4)
sns.scatterplot(data = FS_df.iloc[:10,:], x = FS_df.iloc[:10,:].index, y = 0, markers = 'o', s=300 ) 
plt.legend(['{}_BC'.format(target.name), '{}'.format(target.name)],loc = 4 )
plt.ylabel('R squared', )
plt.show()
final_feat = selected_feat[['TVFA', 'IA/PA', 'pH']]
result, importance, model, compare = pred_result(target, final_feat, xgb_reg, save = True, folder = 'model/fig4')
#####################test dataset################################
Y_test = y_test['HPr']
target_np = Y_test.to_numpy()
Y_test_bc, lambd = stats.boxcox(target_np)
X_test_final = X_test[['TVFA', 'IA/PA', 'pH']]
Y_pred_bc = model.predict(X_test_final)
Y_pred  = inv_boxcox(Y_pred_bc, lambd)
comparison2(Y_pred, Y_test, title = 'Alkalinity prediction (grid search)')
comparison(Y_pred, Y_test, title = 'prediction with XGBRFRegressor (test data)', lambd = None)
print("final model r2 :", model.score(X_test_final, Y_test_bc))



##[Fig4] acetate, propionate prediction with predicted TVFA##
#feature
pred_TVFA = pd.read_excel('fig/fig3 rawdata.xlsx', header = None)
x_test.reset_index(inplace = True, drop = True)
x_test[['TVFA']] = pred_TVFA
X_test= scaler.transform(x_test)
X_test = pd.DataFrame(X_test, columns = features.columns)
features_HPr = X_test[['TVFA', 'IA/PA', 'pH']]
#target
target = y_test['HPr']
target_bc, lambd_target = stats.boxcox(target)
#prediction
model_HPr = joblib.load('model/fig4/XGBRFRegressor_HPr_1.sav')
print("final model r2 :", model_HPr.score(features_HPr, target_bc))
pred_target = model_HPr.predict(features_HPr)
comparison2(pred_target, target_bc, title = 'prediction of [{}] from test dataset'.format(target.name))
pred_result = comparison(pred_target, target_bc, title = 'prediction of [{}] from test dataset'.format(target.name), lambd = lambd_target)


##[Fig5] SHAP value##
model_TVFA= joblib.load('model/fig3/XGBRFRegressor_TVFA_1.sav')
import shap
explainer = shap.TreeExplainer(model_TVFA, )
shap_val = explainer(X_train[['IA/PA', 'FW_HAc']])
    #1) summary plot : mean shap value of each features
shap.summary_plot(shap_val, X_train[['IA/PA', 'FW_HAc']], plot_type = 'bar')
    #2) whether each feature impact positively or negatively on the model output
shap.plots.waterfall(shap_val[12])

    #3) to see each individual feature on the general model output (SHAP determines the color automatically, that has the most irrelevant impact on the feature)
feature = 'IA/PA'
shap.plots.scatter(shap_val[:, 'FW_HAc'], color = shap_val) 
    #*4) to see the positive/negative impact of degree of feature value on SHAP  (RECOMMENDED!)
shap.plots.beeswarm(shap_val)

















model = forest_reg
result, importance, model = pred_result(target, features, model)
target_bc, lambd = stats.boxcox(target)
X_train, X_test, y_train, y_test = split_data(features, target, 0.3)





selected_feat = features.loc[:, sorted_features[0:5]]
result, importance = pred_result(targets, selected_feat, models)

feature_select = pd.DataFrame()
model = forest_reg
for i in range(1,10):
    selected_feat = features.loc[:, sorted_features[0:i]]
    result, _ = pred_result(n_targets, selected_feat, models)
    result.index = [selected_feat.columns[-1]]
    feature_select = pd.concat([feature_select, result])

final_feat = selected_feat.iloc[:,0:4]
X_train, X_test, y_train, y_test = split_data(final_feat, n_targets_bc, 0.3)
result, importance = pred_result(n_targets, final_feat, models, True)

model = joblib.load('model/AdaBoostRegressor_TVFA + EtOH_1.sav')
final_result = model.score(X_test, y_test)



import shap
explainer = shap.TreeExplainer(model, )
shap_val = explainer(X_train)
    #1) summary plot : mean shap value of each features
shap.summary_plot(shap_val, X_train, plot_type = 'bar')
    #2) whether each feature impact positively or negatively on the model output
shap.plots.waterfall(shap_val[0])
    #3) to see each individual feature on the general model output (SHAP determines the color automatically, that has the most irrelevant impact on the feature)
'''
feature = 'Na', 'NH4', 'COD', 'K'
'''
feature = 'IA/PA'
shap.plots.scatter(shap_val[:, feature], color = shap_val) 
    #*4) to see the positive/negative impact of degree of feature value on SHAP  (RECOMMENDED!)
shap.plots.beeswarm(shap_val)
## new features
new_feature_attributes = importance[importance > float(0.01)].index.tolist()
features = features[new_feature_attributes]


test_pred = model.predict(X_test)
comparison2(test_pred, y_test, title = 'Alkalinity prediction (grid search)')
comparison(test_pred, y_test, title = 'Alkalinity prediction (grid search)', lambd = lambd)
print("final model r2 :", model.score(X_test, y_test))
print("feature importances ", model.feature_importances_)
feature_importance_plot(model, features)



model = joblib.load('model/RandomForestRegressor_TVFA_1.sav')
pred_target = model.predict(features[['IA/PA', 'FW_TVFA + EtOH', 'FW_HAc', 'FW_Protein']])
model_HAc = joblib.load('model/RandomForestRegressor_HPr_1.sav')
pred_feat = pd.DataFrame(pred_target, columns = ['TVFA'])
pred_feat = pd.concat([pred_feat, features[['Na', 'TS', 'FW_SO4']]], axis = 1, )
target = model.predict(pred_feat)
target_bc, lambd = stats.boxcox(targets['HAc'])

comparison2(target, target_bc, title = 'Alkalinity prediction (grid search)')
comparison(target, target_bc, title = 'Alkalinity prediction (grid search)', lambd = lambd)
print("final model r2 :", model.score(target, target_bc))
feature_importance_plot(model_HAc, pred_feat)
