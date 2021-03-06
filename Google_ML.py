# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:19:09 2016

@author: Leo
"""

# Regression of Google Stock Price
# Print statements left in but commented out
import pandas as pd
import quandl, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from matplotlib import style
from datetime import datetime

style.use('ggplot')

# Produce initial data frame using example dataset from Quandl
df = quandl.get("WIKI/GOOGL")
df = df[["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume"]]

# Show top five rows of dataset and column headings
#print (df.head())


# Analysis of useful data requires useful columns so refactor data to fit more 
# useful headings
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] *100
df["PCT_Change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] *100

df = df[["Adj. Close","HL_PCT","PCT_Change","Adj. Volume"]]

#print (df.head())

# Set variable so that forecast_col can be different column in future without 
# having to edit algorithm extensively
forecast_col = 'Adj. Close'

# ML cannot work with 'na' fields, must have a value. 
# Assigned large value so that it falls into 'outlier' category of algorithm.
df.fillna(-99999, inplace = True)

# Attempt to predict ahead of df by 1%
# I.e. Use data from 1 day ago to predict today
# math.ceil rounds numbers up to nearest whole number and returns as float
# math.ceil requires math package

forecast_out = int(math.ceil(0.1*len(df)))
#print(forecast_out)

df['Label'] = df[forecast_col].shift(-forecast_out)
# Thus the Adj. Close is now the price close price 1% in the future

#print(df.head())

# Training and Testing
# X = features, y = labels

# Feature is the whole dataframe except the label column and converted into 
# an np array
X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['Label'])
y = np.array(df['Label'])

#print(len(X), len(y))

# cross_validation takes data and shuffles it up as well as splitting it into 
# test and train sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
test_size=0.2)

# choose a classifier - for regression = linear regression
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# test scores ~96% accuracy on predicting stock prices after a 1% time shift
# Means the model predicts the price in 30 days to 96% accuracy
#print(accuracy)

#This is the key line! The crux of prediction comes from this code
forecast_set = clf.predict(X_lately)

# Print the next 30 days (1% of df) of unknown prices, with accuracy and 
# the number of days forecast
print(forecast_set, accuracy, forecast_out)

df ['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range (len(df.columns)-1)] + [i]
    
df ['Adj. Close'].plot()
df ['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel ('Date')
plt.ylabel ('Price')
plt.show()