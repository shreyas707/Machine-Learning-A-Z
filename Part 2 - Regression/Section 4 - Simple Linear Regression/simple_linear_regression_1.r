# Simple Linear Regression

# Importing DataSet
dataset = read.csv('Salary_Data.csv')

# Splitting dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# We don't need feature scaling here as simple linear regression in R takes care of it.
# Fitting simple linear regression to Training Set

# Salary ~ YearsExperience means Salary is proportional to YearsExperience
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# To get some information about simple linear model, in console type summary(regressor) 
# In the summary, in coefficients, if there are no stars, it means no statistical significance. If there are 3 stars, it means that the model has high statistical significance
# P value is another indicator of statistical significance. Lower the p value, the higher the impact of independent variable on the dependent variable. Usually, if p value is lower than 5%, the independent variable is highly significant. Anything lower is less significant.

# Predicting Test Set results
y_pred = predict(regressor, newdata = test_set)

# Visualising the Training Set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  # plot observation points
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  # since we want to take predicted salaries, y cannot be of training_set
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualising the Test Set results
ggplot() +
  # plot observation points
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  # since we want to take predicted salaries, y cannot be of training_set
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')