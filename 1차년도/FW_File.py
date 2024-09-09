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


path = 'E:/OneDrive - postech.ac.kr/연구/VFA prediction/data/Overall.xlsx'
data = pd.read_excel(path, sheet_name = 0, index_col = 'Sample #')

#HBu,HVa, HCa
BuVaCa = data.columns[21:27]
for n,i in enumerate(BuVaCa):
    if 'i' in i:
        data['H{}'.format(i[-2:])] = data[i] + data[BuVaCa[n+1]]
data = data.drop(BuVaCa, axis = 1)        

#Add substrate char. as input variables
cols = data.columns[3:]
subcol = []
for item in cols:
    subcol.append('FW_{}'.format(item))
substrate = pd.DataFrame(columns = subcol)
data = data.append(substrate)

for i in range(0, len(data)):
    if 'M' in data.iloc[i, 2]:
        fw = np.where((data['Sample'].str.contains('F')) & (data['Site'] == data.iloc[i,1]) & (data['Round'] == data.iloc[i,0]))[0][0]
        data.loc[i+1, subcol] = data.iloc[fw,3:37].values

data.to_excel('Finalv1.xlsx')
