#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:05:11 2018

@author: shreyas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# [:, 1] will make X a vector of features instead of matrix. [:, 1:2] will make it matrix of featuers
# With vector, size was (10,) which means, 10 lines and nothing else. With matrix, size is (10,1) which means 10 lines and 1 column
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# We're not creating a training set and a test set because the data is too low and because we need to make very accurate prediction.
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Building both Linear and Polynomial Regression models to compare both

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# More the degree, better the results
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression Model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Model
# Right now, since there are only 10 levels, it is plotting straight lines between predictions. To make it look like a proper curve, instead of incrementing X by 1, increment X by 0.1
# arange(lowerbound, upperbound, incremental_value)
X_grid = np.arange(min(X), max(X), 0.1)
# reshape(number_of_lines, number_of_columns)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regerssion
lin_reg.predict(6.5)

# Predicting a new resultw ith Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))