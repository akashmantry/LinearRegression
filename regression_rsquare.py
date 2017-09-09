#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:25:11 2017

@author: akashmantry
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_dataset(number_of_points, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(number_of_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    

def best_fit_slope_and_intercept(xs, ys):
    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    b = mean(ys) - m*mean(xs)
    return m, b

# sum of square of difference between y_original and y_line
def squared_error(ys_original, ys_line):
    return sum((ys_original-ys_line)**2)

# accuracy or confidence
# r(square) = 1 - ((SE y(hat))/(SE(y(bar))))
def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regression_line = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    
    return 1 - (squared_error_regression_line/squared_error_y_mean)


xs, ys = create_dataset(40, 40, 2, correlation='pos')
m, b = best_fit_slope_and_intercept(xs, ys)

# y = mx + b
regression_line = [m*x + b for x in xs]

r_sqaured = coefficient_of_determination(ys, regression_line)
print(r_sqaured)

plt.scatter(xs,ys,color='#003F72', label = 'data')
plt.plot(xs, regression_line, label = 'regression line')
plt.legend(loc=4)
plt.show()



