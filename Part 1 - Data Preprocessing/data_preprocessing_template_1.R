# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
# taking column age, 'is.na' tells if a value is missing
# mean is an inbuilt function. na.rm = TRUE includes x even if value is na
# third parameter is the else part of the program (what to do if value is not missing)
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
# taking cloumn salary
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Encoding categorical data
# we don't categorize it like in python because r works
dataset$Country = factor(dataset$Country, levels = c('France', 'Spain', 'Germany'), labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased, levels = c('Yes', 'No'), labels = c(1, 0))

# Splitting the dataset into the Training set and Test set
# run this once to install the package => install.packages('caTools')
# to use the library run => library(caTools)
# set.seed is like random_state in python
library(caTools)
set.seed(123)
# sample.split returns TRUE for training set and FALSE for test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# exclude feature scaling to countries and purchased columnds in training_set because they are of type factor (line 17) and not numeric.
# [, 2:3] = taking only columns 2 and 3 (age and salary)
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = subset(test_set[, 2:3])
