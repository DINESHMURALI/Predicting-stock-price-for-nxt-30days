# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:31:19 2019

@author: DINESH
"""

!pip install quandl

import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df = quandl.get("WIKI/AMZN")
print(df.head())

df = df[['Adj. Close']]
print(df.head())

forecast_out = 30
df['prediction']=df[['Adj. Close']].shift(-1)
print(df.tail())

## create the independent data set (x)
# convert the dataframe to a numpy array
x = np.array(df.drop(['prediction'],1))

#remove the last 'n' rows
x = x[:-forecast_out]
print(x)

#create the independent data set (y)
#convert the dataframe into nmpy array 
y = np.array(df['prediction'])
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#create and train in support vector machine machine (regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

svm_conf = svr_rbf.score(x_test, y_test)
print('svr_confidence:', svm_conf)

# create and train in linear reg model
lr = LinearRegression()
lr.fit(x_train, y_train)

lr_conf = lr.score(x_test, y_test)
print('lr_confidence:', lr_conf)

# set _x_forecast equal to the last 30 rows of the original data set from Adj.close cplumn
x_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:]
print(x_forecast)

lr_pred = lr.predict(x_forecast)
print(lr_pred)

svr_pred = svr_rbf.predict(x_forecast)
print(svr_pred)

