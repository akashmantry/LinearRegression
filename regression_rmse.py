#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:49:36 2017

@author: akashmantry
"""

import numpy as np
import pandas as pd
from random import seed
from random import randrange


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    cross_validation_dataset = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset_copy)/n_folds)
    
    for i in range(0, n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        cross_validation_dataset.append(fold)
    return cross_validation_dataset

# Calculate root mean squared error - metric to determine model performance
def root_mean_squared_error(actual_y, predicted_y):
    total_error = 0.0
    for i in range(0, len(actual_y)):
        error = predicted_y[i] - actual_y[i]
        total_error += error**2
    mean_error = total_error/float(len(actual_y))
    return mean_error

def predict(row, coefficients):
    for i in range(0, len(row)-1):
        y_hat = coefficients[i+1] * row[i] + coefficients[0]
    return y_hat

# Estimate linear regression coefficents using stochastic linear regression
"""
    epoch - number of times to run through training dataset and update coefficients
    learning_rate - how fast/slow the coefficients are determined to acheive convergence
    
    For each row in training set, predict the value and get the error.
    Then use the equation to update the coefficients
"""
def determine_sgd_coefficients(training_set, n_epoch, learning_rate):
    coefficients = [0.0 for _ in range(0, len(training_set[0]))]
    for _ in range(0, n_epoch):
        for row in training_set:
            y_hat = predict(row, coefficients)
            error = y_hat - row[-1]
            coefficients[0] = coefficients[0] - learning_rate * error
            for i in range(0, len(row)-1):
                coefficients[i+1] = coefficients[i+1] - learning_rate * error * row[i]
    return coefficients
"""
    Evaluate algorithm's performance using a cross validation split.
    Make n_fold partition of train set.
    For each fold, make it a test set and others as train set and retrieve
    the root mean squared error and store it.
"""
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		rmse = root_mean_squared_error(actual, predicted)
		scores.append(rmse)
	return scores

def linear_regression_sgd(train, test, learning_rate, n_epoch):
    predictions = list()
    coefficients = determine_sgd_coefficients(train, n_epoch, learning_rate)
    for row in test:
        y_hat = predict(row, coefficients)
        predictions.append(y_hat)
    return predictions

# Importing the dataset
dataset = pd.read_csv('winequality-white.csv')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)

dataset = dataset.tolist()

n_folds = 5
learning_rate = 0.01
n_epoch = 50
scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, learning_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))





