#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 08:43:41 2017

@author: shreyas
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# importing independent variable 'x'
# : = all rows, :-1 = except last column, take all rows except last column
X = dataset.iloc[:, :-1].values
# importing dependent variable vector 'y'
# : = all rows, 1 = second column as index starts from 0
y = dataset.iloc[:, 1].values
# X size is (30, 1) but y is (30,). It tells that X is matrix that has 1 column and since y doesn't have anything, its a vector.
# X is the matrix of (feature) independent variable and y is a vector of dependent variable

# Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
# Usually test set !> 0.4. test set + train set = 1. random_state is not required
# Here test set is 1/3 the size of the data and training set would be 2/3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Simple Linear Regression in Python takes care of feature scaling, so we don't have to worry about that.

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fitting regressor object to training set
# fits the line and returns the slope
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualising the training set results
# observation points (real values)
plt.scatter(X_train, y_train, color = 'red')
# regression line
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
# since regressor is trained on the training set, regression line is already determined
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()