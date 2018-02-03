#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:49:36 2017

@author: shreyas
"""
# numpy contain mathematical tools
import numpy as np
# used to plot charts
import matplotlib.pyplot as plt
# this is to import dataset (csv)
import pandas as pd
import pdb

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# iloc returns the columns
# : = all rows, :-1 = take all columns except last
X = dataset.iloc[:, :-1].values
# importing dependent variable vector 'y'
# column index starts from 0 in python. taking only the last(4th) column
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# missing_values = 'NaN' checks for values that are NaN, strategy = "mean" is a default parameter. axis = 0 takes column
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
# fit imputer object to matrix X
# : = all rows, 1:3 = columns 1 & 2
imputer = imputer.fit(X[:, 1:3])
# transform replaces missing data
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# creating an object of LabelEncoder class
labelencoder_X = LabelEncoder()
# : = all rows, 0 = first column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# OneHotEncoder - Here, there's France, Germany and Spain. So 3 columns are created (alphabetical order) and in the respective column, 1 is added. The other values are 0. We need this because labelEncoder gives random values for categories and ml will think those values have significance.
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# since y is a dependent variable, OneHotEncoder is not needed
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Usually test_size !> 0.4. test_size + train_size = 1. random_state is not needed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# This is only for my reference to compare values after feature scaling
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Feature scaling is required because for example, when we take age and salary, they are very far apart. To bring the values on the same scale and close to each other, we use feature scaling.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)