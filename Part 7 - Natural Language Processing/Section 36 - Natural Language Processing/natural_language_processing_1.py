#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 08:35:14 2018

@author: shreyas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# quoting = 3 ignores any double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# CLeaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# corpus is a collection of text in ml terms
corpus = []
# range(0, 1000) - 0 is the lower bound, 1000 is the upper bound
stop_words = set(stopwords.words('english'))
for i in range(0, 1000):
    # dataset['Review'][0] gives the first review
    # remove anything that is not a-z or A-Z. ' ' (second parameter) is so that spaces are not removed making the words stick together
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Making all the words lowercase
    review = review.lower()
    review = review.split()
    # PorterStemmer is to stem the words. That is, to change the words to their original form. Eg: loved becomes love
    ps = PorterStemmer()
    # iterating through review and checking if any word from stopwords exists. if it exists, then remove it. finally stem all the words
    # Note: if the sentence is big, use set as algorithms in python work faster when going through words in set than list
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# X is a sparse matrix (matrix with mostly 0s is a sparse matrix). It is the independent variable
# X has to be fit to the corpus, to analyze it and to look at all the words, look at how to apply different parameters we input in count vectorizer
X = cv.fit_transform(corpus).toarray()

# y is the dependent variable
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework - Calculating Accuracy, Precision, Recall and F1 Score
true_positive = cm[1][1]
false_positive = cm[1][0]

true_negative = cm[0][0]
false_negative = cm[0][1]

accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

precision = true_positive / (true_positive + false_positive)

recall = true_positive / (true_positive + false_negative)

f1_score = 2 * precision * recall / (precision + recall)