# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:47:41 2021

@author: sujin
"""

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from functions_2 import comparison,display_scores, comparison2, rmse, mape, split_data,rmse_inv, ScoringAUC, plot_learning_curves
from sklearn.inspection import permutation_importance
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from imblearn.metrics import macro_averaged_mean_absolute_error

class reg:
    def __init__(self, model, X_train, y_train, lambd ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.lambd = lambd
    def model_fit(self, x, y, early_stopping = False):
        if early_stopping == False:
            self.model.fit(x, y)
        else:
            X_train, X_val, y_train, y_val = train_test_split(x,y, random_state = 42)
            self.model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], early_stopping_rounds = 20,)  
            return X_train, X_val, y_train, y_val 
    def predict(self, x, y):
        pred = self.model.predict(x)
        rmse(y, pred)
        mape(y, pred)
        compare = comparison(pred, y, title = 'prediction with {}'.format(str(self.model).split('(')[0]), lambd = self.lambd)
        comparison2(pred, y, title = 'prediction with {}'.format(str(self.model).split('(')[0]))
        plt.scatter(compare.actual_value, compare.error)
        plt.ylim([0,100])
        plt.ylabel('error(%)')
        plt.xlabel('actual value')
        plt.show()
        return compare, pred
    def cross_val(self, x, y):
        score = display_scores(self.model, x, y)
        return score
    def permutation_importance_plot(self, X, y):
        result = permutation_importance(self.model, X, y, n_repeats= 20, random_state = 20, n_jobs = -1)
        importances = pd.Series(result.importances_mean, index = X.columns)
        importances.sort_values(inplace = True)
        fig, ax = plt.subplots()
        importances.plot.bar(yerr = result.importances_std, ax = ax)
        ax.set_title("feature importances using permutation on model")
        ax.set_ylabel('Mean accuracy decrease')
        fig.tight_layout()
        plt.show()
        return importances

class clf:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
    def model_fit(self, early_stopping = False):
        if early_stopping == False:
            self.model.fit(self.X_train, self.y_train)
        else:
            X, X_val, y, y_val = train_test_split(self.X_train, self.y_train, random_state = 42)
            self.model.fit(X, y, eval_set = [(X,y), (X_val, y_val)], 
                           early_stopping_rounds = 20)
    def predict(self):
        pred = self.model.predict(self.X_train)
        y_score = self.model.predict_proba(self.X_train)
        accuracy = roc_auc_score(self.y_train, y_score, multi_class = 'ovo')
        #print("accuracy_score : ",accuracy)
        macro_mae = macro_averaged_mean_absolute_error(self.y_train, pred)
        print('macro mae :', macro_mae)
        report = classification_report(self.y_train, pred, labels = list(range(len(np.unique(self.y_train)))))
        print(report)
        conf_matrix = confusion_matrix(self.y_train, pred)
        fig, ax = plot_confusion_matrix(conf_mat = conf_matrix, figsize = (6,6), cmap = plt.cm.Greens)       
        plt.xlabel('Predictions', fontsize = 18)
        plt.ylabel('Actuals', fontsize = 18)
        plt.title('Confusion Matrix', fontsize = 18)
        plt.show()
        return pred, accuracy, report
    def cross_val(self):
        kfold = StratifiedKFold(n_splits = 5,shuffle = True, random_state = 10)
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv = kfold, scoring = ScoringAUC())
        print("scores of ", str(self.model))
        print("score: ", scores)
        print("mean: ", scores.mean())
        print("std value: ", scores.std())
        return kfold, scores
    def cross_val_conf_mx(self, kfold):
        pred = cross_val_predict(self.model, self.X_train, self.y_train, cv = kfold)
        conf_matrix = confusion_matrix(self.y_train, pred)
        fig, ax = plot_confusion_matrix(conf_mat = conf_matrix, figsize = (6,6), cmap = plt.cm.Greens)       
        plt.xlabel('Predictions', fontsize = 18)
        plt.ylabel('Actuals', fontsize = 18)
        plt.title('Confusion Matrix', fontsize = 18)
        plt.show()
    def plot_learning_curve(self, ):
        title = 'Learning Curves'
        cv = ShuffleSplit(n_splits=30, test_size=0.3, random_state=0)
        plot_learning_curves(self.model, title, self.X_train, self.y_train, ylim=(0.7, 1.01),cv=cv, n_jobs=4)
        plt.show()
    def plot_learning_curves(self, ):
        X, X_val, y, y_val = train_test_split(self.X_train, self.y_train, stratify = self.y_train, random_state = 42)
        train_errors, val_errors = [],[]
        for m in range(10, len(y)):
            self.model.fit(X[:m],y[:m])
            y_train_predict = self.model.predict_proba(X[:m])
            y_val_predict = self.model.predict_proba(X_val)
            train_accuracy = roc_auc_score(y[:m], y_train_predict, multi_class = 'ovo')
            val_accuracy = roc_auc_score(y_val, y_val_predict, multi_class = 'ovo')
            train_errors.append(train_accuracy)
            val_errors.append(val_accuracy)      
        plt.plot(train_errors,'r-+', linewidth = 2, label = 'training set')
        plt.plot(val_errors, 'b-', linewidth = 3, label = 'validation set')
        plt.legend()
        plt.xlabel('size of training set')
        plt.ylabel('auc')
        plt.title('plot learning curves')
        plt.show()
    

import tensorflow as tf

def build_model(input_shape, n_hidden=10, n_neurons = 30, learning_rate = 0.01):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    
    model.compile(loss = 'mse', optimizer = optimizer)
    return model


