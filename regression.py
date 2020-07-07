# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:31:07 2020

@author: popo
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('国华20200319.csv')

label_columns = list(filter(lambda x: 'F_S' in x, df.columns))
input_columns = list(filter(lambda x: 'F_S' not in x, df.columns))
X = df.loc[:, input_columns]
X = X.iloc[:, 1:]
y = df.loc[:, label_columns]
y = y.iloc[:, 0] + y.iloc[:, 1]

for i in range(len(X.columns)):
    X.iloc[:, i: i+1] = MinMaxScaler().fit_transform(X.iloc[:, i: i+1])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor().fit(X_train, y_train)
out = model.predict(X_test)
mean_squared_error(out, y_test)


