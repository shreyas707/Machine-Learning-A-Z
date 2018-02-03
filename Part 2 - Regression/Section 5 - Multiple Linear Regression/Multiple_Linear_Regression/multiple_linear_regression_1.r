# Muktiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to Training Set
# . can be replaced by 'R.D.Spend + Administration + Marketing.Spend + State'. space is replaced by '.' in R. Hence R.D.Spend and not R D Spend
# ~ - linear combination, . - all independent variables
regressor = lm(formula = Profit ~ .,
               data = training_set)
# type summary(regressor) in console to see the statistical significance of variables
# note that R removes the dummy variables by itself (in summary, it shows only 2 states as the 3rd one is dummy. This is done manually in python)

# Predicting the Test Set results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination
# We're taking the whole dataset but we can use training set too.
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)