# LinearRegression
Goal of this project is to implement Linear Regression using Gradient Descent algorithm and Normal Equations without using any Machine learning libraries.

# Gradient descent for Linear Regression

I used the white-wine dataset where the task is to predict the quality of wineepending on features like fixed acidity, volatile acidity, citric acid, etc. There are a total of 11 features.

I use the following set of parameters:
a) learning rate = 0.01
b) number of epocs = 50
c) k-folds = 5

I use k-fold cross-validation and use root mean square error (RMSE) on each fold to determine the performance of the model. I then output the overall mean. For normalizing the dataset, I standardized features by removing the mean and scaling to unit variance.

# Without gradient descent

I also created a randomly generated dataset and applied linear regression to it to study the impact of size, variance, and positive and negative correlation of data on performance. The metric used was r-square. The dataset consisted of only one feature and the best fit slope and intercept normal equations were used. 
