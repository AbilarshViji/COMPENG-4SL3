
#1-nearest neighbour classification for the iris dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load dataset
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris.DESCR)
X, t = load_iris(return_X_y=True)
#print(t)

#split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/4, random_state = 5)
X_train = X_train[:,2:]
X_test = X_test[:,2:]
M = len(X_test)
N = len(X_train) 
print(M,N)

#plot data for features 2 and 3
#separate training data for different classes
#class 0
i0 = np.asarray(np.nonzero(t_train==0)) #indexes where class is 0
#print(i0)
[m,n] = i0.shape
X_train_0 = np.zeros((n,2))
t_train_0 = np.zeros(n)
#print(t_train_0)
for i in range(n):
    X_train_0[i,:] = X_train[i0[0,i],:] 
#print(X_train_0)

#class 1
i1 = np.asarray(np.nonzero(t_train==1)) #indexes where class is 0
#print(i1)
[m,n] = i1.shape
#print(n)
X_train_1 = np.zeros((n,2))
t_train_1 = np.ones(n)
#print(t_train_1)
for i in range(n):
    X_train_1[i,:] = X_train[i1[0,i],:] 
#print(X_train_1)

#class 2
i2 = np.asarray(np.nonzero(t_train==2)) #indexes where class is 0
print(i2)
[m,n] = i2.shape
print(n)
X_train_2 = np.zeros((n,2))
t_train_2 = np.ones(n)
t_train_2 += 1
#print(t_train_2)
for i in range(n):
    X_train_2[i,:] = X_train[i2[0,i],:] 
#print(X_train_2)
    
#plot classes
plt.scatter(X_train_0[:,0], X_train_0[:,1], color = 'red')
plt.scatter(X_train_1[:,0], X_train_1[:,1], color = 'blue')
plt.scatter(X_train_2[:,0], X_train_2[:,1], color = 'green')
plt.show()

#start classification of test examples
#initialize arrays
dist = np.zeros((M,N)) #2dim array to store distances from test points to trainig points
ind = np.zeros((M,N))  #2dim array to store the order after sorting the distances
u = np.arange(N)       # array of numbers from 0 to N-1
for j in range(M):
    ind[j,:] = u
#print(dist[:5,:])
    
#compute distances and sort
for j in range(M): #each test point
    for i in range(N): #each training point
        #z = X_train[i]-X_valid[j] # just one feature
        z = X_train[i,:]-X_test[j,:]
        dist[j,i] = np.dot(z,z)
        #ind[j,:] = np.argsort(dist[j,:])
#print(dist[:5,:5])
ind = np.argsort(dist)
print(ind.shape)
#print(dist[0,:])
#print(ind[0,:])

# compute predictions and error with 1NN
y = np.zeros(M) # initialize array of predictions
for j in range(M):
    y[j] = t_train[ind[j,0]]
#print(y)
#print(t_test)
z = y - t_test
print(z)
err = np.count_nonzero(z)/M  #mislassification rate
print(err)