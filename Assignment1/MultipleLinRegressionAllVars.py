# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:36:21 2020

@author: Sorina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
#boston = load_boston()
#print(boston.DESCR)

#import data set from scikit
from sklearn.datasets import load_boston
X, t = load_boston(return_X_y=True)
print(X.shape)
N=len(X)  # number of rows
print(N)

# split data into trainig and test sets
from sklearn.model_selection import train_test_split
# X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/4, random_state = 6)
# print(X_train.shape) # X_train is 2D, but y_train is 1D
# print(t_train.shape)
M = len(X) #number rows in test set
N = len(X) #number rows in train set
print(N, M)
t_train, t_test = t, t
# add dummy in trainig set
new_col=np.ones(N)
# print(new_col[:8])
X1_train = np.insert(X, 0, new_col, axis=1)
print(X1_train)

# add dummy in test set
new_col=np.ones(M)
# print(new_col[:8])
X1_test = np.insert(X, 0, new_col, axis=1)
# print(X1_test)

# perform linear rgression
A = np.dot(X1_train.T,X1_train)
print("a",A)
A1=np.linalg.inv(A)
#print(A1)
t1=np.dot(X1_train.T,t_train)
print("c", t1)
w = np.dot(A1,t1)
print("w", w)

#compute predictions on trainig set
y_train_predict = np.dot(X1_train,w)
print(X1_train.shape, w.shape)
print(y_train_predict[:8])

#compute training error
diff = np.subtract(t_train, y_train_predict)
err_train = np.dot(diff,diff.T)/N

#compute predictions on test set
y_test_predict = np.dot(X1_test,w)
diff_test = np.subtract(t_test, y_test_predict)
#print(diff_test[:8])

#compute test error
err_test = np.dot(diff_test,diff_test.T)/M
print(err_train, err_test)