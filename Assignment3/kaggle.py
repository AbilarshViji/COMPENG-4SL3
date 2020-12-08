# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('BankNote_Authentication.csv')
data.head()

from sklearn import preprocessing
col_list = ['variance', 'skewness', 'curtosis', 'entropy']
features2 = data.copy()[col_list]

scaler = preprocessing.StandardScaler()
features2_standarized = scaler.fit_transform(features2)

features2_standarized

print('Mean:', round(features2_standarized[:,0].mean()))
print('Standard deviation:', features2_standarized[:,0].std())

data[col_list] = features2_standarized
data.head()

data.tail()

# Pecah data menjadi train dan test dengan proporsi 3:1
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)
train.head()

test.head()

# include only the rows having label = 0 or 1 (binary classification)
X = train[train['class'].isin([0, 1])]

# target variable
Y = train[train['class'].isin([0, 1])]['class']

# remove the label from X
X = X.drop(['class'], axis = 1)

# implementing a sigmoid activation function
def relu(z):
    # s = 1.0/ (1 + np.exp(-z))    
    return np.maximum(0, z)
    # return s

def sigmoid(z):
    s = 1.0/ (1 + np.exp(-z))   
    return s

def network_architecture(X, Y):
    # nodes in input layer
    n_x = X.shape[0] 
    # nodes in hidden layer
    n_h = 2          
    n_h_2 = 2
    # nodes in output layer
    n_y = Y.shape[0]
    return (n_x, n_h, n_h_2, n_y)

def define_network_parameters(n_x, n_h, n_h_2, n_y):
    W1 = np.random.randn(n_h,n_x) * 0.01 # random initialization
    b1 = np.zeros((n_h, 1)) # zero initialization
    W2 = np.random.randn(n_h_2, n_h) * 0.01
    b2 = np.zeros((n_h_2, 1))
    W3 = np.random.randn(n_y,n_h_2) * 0.01 
    b3 = np.zeros((n_y, 1)) 
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3} 

def forward_propagation(X, params):
    Z1 = np.dot(params['W1'], X)+params['b1']
    A1 = relu(Z1)

    Z2 = np.dot(params['W2'], A1)+params['b2']
    A2 = relu(Z2)

    Z3 = np.dot(params['W3'], A2)+params['b3']
    A3 = relu(Z3)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}  

def compute_error(Predicted, Actual):
    logprobs = np.multiply(np.log(Predicted), Actual)+ np.multiply(np.log(1-Predicted), 1-Actual)
    cost = -np.sum(logprobs) / Actual.shape[1] 
    return np.squeeze(cost)

def backward_propagation(params, activations, X, Y):
    m = X.shape[1]
    # print(m)
    # output layer
    dZ3 = activations['A3'] - Y # compute the error derivative 
    dW3 = np.dot(dZ3, activations['A2'].T) / m # compute the weight derivative 
    db3 = np.sum(dZ3, axis=1, keepdims=True)/m # compute the bias derivative
    
    # hidden layer
    dZ2 = np.dot(params['W3'].T, dZ3)*(1-np.power(activations['A2'], 2))
    dW2 = np.dot(dZ3, dZ2.T)/m
    db2 = np.sum(dZ2, axis=1,keepdims=True)/m
    # print(dZ3.shape, dW3.shape, dZ2.shape, dW2.shape, params['W3'].shape, params['W2'].shape)
    # hidden layer
    dZ1 = np.dot(params['W2'].T, dZ2)*(1-np.power(activations['A1'], 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1,keepdims=True)/m

    # # output layer
    # dZ2 = activations['A2'] - Y # compute the error derivative 
    # dW2 = np.dot(dZ2, activations['A1'].T) / m # compute the weight derivative 
    # db2 = np.sum(dZ2, axis=1, keepdims=True)/m # compute the bias derivative
    
    # # hidden layer
    # dZ1 = np.dot(params['W2'].T, dZ2)*(1-np.power(activations['A1'], 2))
    # dW1 = np.dot(dZ1, X.T)/m
    # db1 = np.sum(dZ1, axis=1,keepdims=True)/m
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

def update_parameters(params, derivatives, alpha = 1):
#def update_parameters(params, derivatives, alpha = 0.4):
    # alpha is the model's learning rate 
    params['W1'] = params['W1'] - alpha * derivatives['dW1']
    params['b1'] = params['b1'] - alpha * derivatives['db1']
    params['W2'] = params['W2'] - alpha * derivatives['dW2']
    params['b2'] = params['b2'] - alpha * derivatives['db2']
    params['W3'] = params['W3'] - alpha * derivatives['dW3']
    params['b3'] = params['b3'] - alpha * derivatives['db3']
    return params

def neural_network(X, Y, n_h, num_iterations=100):
    n_x = network_architecture(X, Y)[0]
    n_y = network_architecture(X, Y)[2]
    
    params = define_network_parameters(n_x, n_h, n_h+1, n_y)
    for i in range(0, num_iterations):
        results = forward_propagation(X, params)
        error = compute_error(results['A3'], Y)
        # print(error)
        # print(results['A3'])
        derivatives = backward_propagation(params, results, X, Y) 
        params = update_parameters(params, derivatives)    
#     print(results)
    # print(error)
#     print(derivatives)
#     print(params)
    return params

def predict(parameters, X):
    results = forward_propagation(X, parameters)
    # print (results['A3'][0])
    # predictions = np.around(results['A3'])    
    predictions = results['A3']
    return predictions

# np.random.seed(2973360718)
y = Y.values.reshape(1, Y.size)
x = X.T.values
model = neural_network(x, y, n_h = 2, num_iterations = 100)

predictions = predict(model, x)
# print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')
print ('Accuracy: ',  ((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) , '%')
# model = neural_network(x, y, n_h = 2, num_iterations = 50)

# predictions = predict(model, x)
# print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')

# model = neural_network(x, y, n_h = 2, num_iterations = 500)

# predictions = predict(model, x)
# print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')
# print (float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100))

# list_k = list( range(10,1000,20))
# print(list_k)

# acc = []
# list_k1 = list( range(1,500,100))
# for nn in list_k:
#     model = neural_network(x, y, n_h = 10, num_iterations = nn)
#     predictions = predict(model, x)
#     acc.append(float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100))

# print(acc)

# import matplotlib.pyplot as plt
# plt.plot( list_k, acc )
# plt.xlabel("Epoch")
# plt.ylabel("Akurasi")
# plt.show()

# max_value = max(acc)
# max_index = acc.index(max_value)

# print('Akurasi terbesar = ' + str(max_value))
# #print(max_index)
# print('Terdapat pada nilai epoch sebesar ' + str(list_k[max_index]))