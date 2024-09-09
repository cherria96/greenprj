# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:57:41 2021

@author: sujin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.special import inv_boxcox
from sklearn.metrics import r2_score

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

def split_train_test(data, test_ratio):
    
    np.random.seed(204)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

class Temp_cat(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        Temp = X["Temp"]
        Temp = pd.cut(Temp, bins = [0,40, np.inf], labels = [0,1])
        return X

pipeline = Pipeline([
    ("temp_cat", Temp_cat()),
    ("std_scaler", StandardScaler())
    ])

def split_data(features, target, test_size = 0.2):
    x_train, x_test, y_train, y_test =train_test_split(features, target, test_size=test_size, random_state= 42)
    standardscaler= StandardScaler()
    X_train = standardscaler.fit_transform(x_train)
    X_test = standardscaler.transform(x_test)
    X_train = pd.DataFrame(X_train, columns = features.columns)
    X_test = pd.DataFrame(X_test, columns = features.columns)
    y_train.reset_index(inplace = True, drop = True)
    y_test.reset_index(inplace = True, drop = True)

    return X_train, X_test, y_train, y_test, x_train, x_test, standardscaler

def scaling(features):
    standardscaler= StandardScaler()
    scaled_features = standardscaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns = features.columns)
    return scaled_features

class ScoringAUC():
    """Score AUC for multiclass problems.
    Average of one against all.
    """
    def __call__(self, clf, X, y, **kwargs):
        from sklearn.metrics import roc_auc_score

        # Generate predictions
        if hasattr(clf, 'decision_function'):
            y_pred = clf.decision_function(X)
        elif hasattr(clf, 'predict_proba'):
            y_pred = clf.predict_proba(X)
        else:
            y_pred = clf.predict(X)

        # score
        classes = set(y)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        _score = list()
        for ii, this_class in enumerate(classes):
            _score.append(roc_auc_score(y == this_class,
                                        y_pred[:, ii]))
            if (ii == 0) and (len(classes) == 2):
                _score[0] = 1. - _score[0]
                break
        return np.mean(_score, axis=0)


def display_scores(model, X, y):
    scores = cross_val_score(model, X, y, scoring = "r2", cv = 5)
    #scores = np.sqrt(-scores)
    print("scores of ", str(model))
    print("score: ", scores)
    print("mean: ", scores.mean())
    print("std value: ", scores.std())
    print()
    return scores.mean()
 
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("mape :", np.mean(np.abs((y_true - y_pred) / y_true)) * 100, "\n")

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print("rmse score: ", rmse, "\n")

def comparison (y_pred, y_true, title, lambd):
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
    if bool(lambd) == True:
        y_pred = inv_boxcox(y_pred, lambd)
        y_true = inv_boxcox(y_true, lambd)
    comparison = pd.DataFrame(y_pred, columns= ['prediction_value'] )
    comparison['actual_value'] = pd.DataFrame(y_true)
    comparison['error'] = (abs(comparison['prediction_value']-comparison['actual_value']))/comparison['actual_value'] * 100
    fig, ax1 = plt.subplots(figsize = (12,8))
    ax1.plot(comparison.index, comparison.prediction_value, 'b')
    ax1.scatter(comparison.index, comparison.actual_value, color = 'red')
    ax1.legend(['prediction', 'actual'], loc= 2)
    #ax2 = ax1.twinx()
    #ax2.bar(comparison.index, comparison.error, color = 'skyblue', alpha = 0.5 )
    #ax2.set_ylabel('error')
    #ax2.set_ylim([0,200])
    #ax2.legend(['error'])
    plt.title(title)
    plt.show()
    
    return comparison


def plot_learning_curves(model, X, y, metric = 'r2'):
    
    '''training plot along with the size of training set'''

    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3, random_state= 42)
    #standardscaler = StandardScaler()
    #X_train = standardscaler.fit_transform(x_train)
    #X_val = standardscaler.transform(x_val)
    train_errors, val_errors = [],[]
    y_val_predict = []
    for m in range(3, len(X_train)):
        model.fit(X_train[:m],y_train[:m])

        y_train_predict = model.predict(X_train[:m])
        y_val_predict.append(model.predict(X_val))
        #train_errors.append(r2_score(model, y_train[:m], y_train_predict))
        #val_errors.append(r2_score(model, y_val, y_val_predict))
    
    plt.plot(train_errors,'r-+', linewidth = 2, label = 'training set')
    plt.plot(val_errors, 'b-', linewidth = 3, label = 'validation set')
    plt.legend()
    plt.xlabel('size of training set')
    plt.ylabel(metric)
    plt.title('plot learning curves')
    plt.show()
    return y_val, y_val_predict
def comparison2 (y_pred, y_true, title):
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
    x = np.arange(min(float(y_pred.min()), float(y_true.min())), max(float(y_pred.max()), float(y_true.max()))*1.5)
    y = x
    plt.figure(figsize = (8,8))
    #plt.fill([0,0,1.4,1.4],[0,1.4,1.4,0],color = 'lightgray', alpha = 0.5)
    #plt.fill([1.4,1.4,9,9],[1.4,10,10,1.4],color = 'pink', alpha = 0.5)
    #plt.text(4,7,'above 1.4', fontsize = 14,)
    #plt.text(2,0.8, 'below 1.4', fontsize = 14)
    plt.scatter(comparison[['actual_value']], comparison[['prediction_value']], color = 'red', alpha = 0.7)
    plt.plot(x,y)
    y1 = x*1.1
    y2 = x*0.9
    y3 = x*1.2
    y4 = x*0.8
    #plt.plot(x, y1, 'lightgray', linewidth = 0.2, alpha = 0.8 )
    #plt.plot(x, y2, 'lightgray', alpha = 0.8, linewidth = 0.2)
    #plt.plot(x, y3, 'lightgray', alpha = 0.4, linewidth = 0.2 )
    #plt.plot(x, y4, 'lightgray', alpha = 0.4, linewidth = 0.2)
   
    #plt.fill_between(x[:], y1[:], y2[:], color = 'lightgray', alpha = 0.8)
    #plt.fill_between(x[:], y3[:], y4[:], color = 'lightgray', alpha = 0.4)

    plt.xlabel('actual_value')
    plt.ylabel('prediction_value')
    plt.xlim( right = x.max())

    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_history(history):
    '''
    Parameter : history
    returns the plot of the history from model trained 
    '''
    import matplotlib.pyplot as plt
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.xlabel("epochs",)
    plt.ylabel("loss", )
    plt.show()

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))
    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label = name_param_2+':'+str(val))
    ax.set_title('grid search scores', fontsize= 20, fontweight = 'bold')
    ax.set_xlabel(name_param_1, fontsize = 16)
    ax.set_ylabel('CV average score', fontsize = 16)
    ax.legend(loc = 'best', fontsize = 15)
    ax.grid('on')
    
def rmse_inv(y_pred, y_true):
    return print('rmse score :' , -np.sqrt(np.mean(np.square(inv_boxcox(y_pred) - inv_boxcox(y_true)))))

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
