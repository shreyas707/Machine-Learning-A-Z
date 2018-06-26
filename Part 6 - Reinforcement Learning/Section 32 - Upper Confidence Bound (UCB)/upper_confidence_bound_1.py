# Upper Confidence Bound

# Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
# n is the number of users
N = 10000
# d is the number of types of ads
d = 10
# Vector[list/array] of 0 of size d
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(numbers_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / sum_of_selections[i]
            # since index starts at 0, log 0 = 1. hence we take n+1 
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            # 1e400 = 10^400           
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1