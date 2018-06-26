# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# Encoding France, Spain and Germany to 0, 1 and 2
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
# Encoding Male and Female to 1 and 0
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Categorical variables are not ordinal i.e., France !> Germany or France etc
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# This is to remove the first column as there are only 3 cities, we'll need 2 columns [00, 01, 10]
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# add is used to add hidden layer
# output_dim - the number of nodes to add in the hidden layer
# tip for output_dim - average of number of nodes in input and output layer
# step 1 of stochastic gradient descent - init - used to randomly initialize the weights close to 0
# Rectifier activation function for hidden layers and sigmoid activation function for output layers
# relu is the rectifier activation function
# input_dim = number of nodes in the input layer (number of independent variables)
classifier.add(Dense(output_dim = 6, init = "uniform", activation='relu', input_dim = 11))

# Adding second input layer
# input_dim is needed only when no layers are created. since this is the second hidden layer, we don't need it
classifier.add(Dense(output_dim = 6, init = "uniform", activation='relu'))

# Adding the output layer
# output layer will have only 1 node as dependent variable is a categorical variable (0 or 1), hence output_dim = 1
# if dependent variable has more than 2 categories, then change output_dim to number of classes and change the activation function to suftmax. It is a sigmoid function for dependent variable with more than 2 categories.
# activation function is sigmoid for output_layer because we need probabilities
classifier.add(Dense(output_dim = 1, init = "uniform", activation='sigmoid'))

# Compiling the ANN (applying stochastic gradient descent on ANN)
# We have only initialized the weights. We need to apply an algorithm to find the best weight to make the neural network powerful. Adding stochastic gradient descent algorithm called adam.
# loss corresponds to a math function. We need this to optimize weights in math. We use a logarithmic loss function because output layer is using sigmoid function
# Since output is binary, we use a loss function called binary_crossentropy. For dependent variables with more than 2 categories, use categorical_crossentropy loss function
# metrics is a criterion used to evaluate a model. Typically accuracy criterion is used. After updating the weights, the algorithm uses accuracy criterion to improve model's performance. Metrics expects a list
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training Set
# We can choose to update the weights either after each observation passing through ANN or after a batch of observations. 
# batch_size is the number of observations after which we want to update the weights
# epoch is a round when the whole training set is passed through ANN
# batch_size and nb_epoch depends on the data scientist. 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
    
# Part 3 - Making the predictions and evaluating the model
# predict method returns the probability (of customers leaving the bank)
y_pred = classifier.predict(X_test)

# we need the predicted result in form of true or false to create the confusion matrix
# anything below 0.5 is false and above 0.5 is true
y_pred = y_pred > 0.50

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0] + cm[1][1])/ 2000