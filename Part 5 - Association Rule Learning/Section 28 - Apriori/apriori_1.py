# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the datset
# header = None removes the titles
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
# lower range 0 is included, upper range 7501 is excluded
for i in range(0, 7501):
    # for every transaction, we're creating a list of all the products of that transaction and appending it to transactions list.
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori

# for min_support, we're considering products that have been purchased at least 3 times in a day. so formula for min_support is 3*7/7500. 3 times a day times 7 days(1 week) divided by total number of products. => 0.0028 ~ 0.003
# min_confidence = 0.2 means that the rules must be correct 20% of the time
# if we get rules with lift higher than 3, the ruels are good. lift is a great insight of the relevance and strength of the rule
# min_length = 2 means that we want minimum of 2 products in the basket. Because having 1 product doesn't make sense for association rule
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)