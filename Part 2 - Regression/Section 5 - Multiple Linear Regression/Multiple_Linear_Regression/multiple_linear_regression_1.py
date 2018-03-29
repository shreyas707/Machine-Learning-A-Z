#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:04:12 2018

@author: shreyas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
# Don't really have to do it manually in python
# : - all rows, 1: - all columns starting from index 1 (without first column)
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# No need of Feature Scaling either because Multiple Linear Regression takes care of it

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(X_test)

# Building the model with Backward Elimination
import statsmodels.formula.api as sm
# From the equation y = b0 + b1*x1 + .. + bn*xn, there is no x0 coefficient for b0. But we can consider x0 = 1 so that value doesn't change. Statsmodel doesn't take into account b0 constant. We will have to add x0 = 1 in the matrix of features (X). In most libraries like linear_model, this is already taken care of. Therefore, we need to add column of 1s in the matrix of features (X) in the start.
# (50, 1) -> 50 lines of 1 column, axis = 1 -> column (0 for row)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# X_opt contains only statistically significant independent variables that affect the profit
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Step 2 in Backward Elimination - Fit full model with all possible predictors (independent variables)
# OLS - Ordinarily Square Algorithm, endog - dependent variable, exog - independent variables with x0. In its description, it says intercept is not included, it means that line 49 is not included by default.
# Next, fit OLS Algo to X_opt and y
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Step 3 - Consider the predictor with highest P value
# summary() gives a table containing all statistical metrics that are useful when building a model
regressor_OLS.summary()

# Removing x2 as its P value is higher than 0.05 and is the highest of all
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing x1
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing x2
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing x2
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
