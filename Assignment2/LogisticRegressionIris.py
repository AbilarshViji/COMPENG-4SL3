#logistic regression for iris data set
#binary classification: old classes 0 and 1 are new class 0; old class 2 is new class 1

#importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading the data set
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris.DESCR)
X, t = load_iris(return_X_y=True)

#splitting the data into the trainig and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/4, random_state = 3)

#use only last two features for prediction
X_train = X_train[:,2:]
X_test = X_test[:,2:]

#separate classes
#plot data for features 2 and 3
#separate training data for different classes
#class 0  
i0 = np.asarray(np.nonzero(t_train==0)) #indexes where class is 0
print("i0", i0)
[m,n] = i0.shape
X_train_0 = np.zeros((n,2))
t_train_0 = np.zeros(n)
#print(t_train_0)
for i in range(n):
    X_train_0[i,:] = X_train[i0[0,i],:] 
print(X_train_0)

# #class 1
# i1 = np.asarray(np.nonzero(t_train==1)) #indexes where class is 0
# #print(i1)
# [m,n] = i1.shape
# #print(n)
# X_train_1 = np.zeros((n,2))
# t_train_1 = np.ones(n)
# #print(t_train_1)
# for i in range(n):
#     X_train_1[i,:] = X_train[i1[0,i],:] 
# #print(X_train_1)

# #class 2
# i2 = np.asarray(np.nonzero(t_train==2)) #indexes where class is 0
# print(i2)
# [m,n] = i2.shape
# print(n)
# X_train_2 = np.zeros((n,2))
# t_train_2 = np.ones(n)
# t_train_2 += 1
# #print(t_train_2)
# for i in range(n):
#     X_train_2[i,:] = X_train[i2[0,i],:] 
# #print(X_train_2)
    
# #plot classes
# plt.scatter(X_train_0[:,0], X_train_0[:,1], color = 'red')
# plt.scatter(X_train_1[:,0], X_train_1[:,1], color = 'blue')
# plt.scatter(X_train_2[:,0], X_train_2[:,1], color = 'green')
# plt.title('Iris flower data (training set)')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.savefig('iris.png')
# plt.show()
    
# #convert to binary classification change 2 to 1, 1 to 0
# #tt_train and tt_test store the targets after the convertion
# tt_train = t_train.copy()
# tt_test = t_test.copy()
# M = len(X_test)
# N = len(X_train)
# for i in range(M):
#     if(t_test[i]==1):
#         tt_test[i] = 0
#     elif(t_test[i]==2):
#         tt_test[i] = 1
    
# for i in range(N):
#     if(t_train[i]==1):
#         tt_train[i] = 0
#     elif(t_train[i]==2):
#         tt_train[i] = 1
# print(t_train)
# print(tt_train)    
        
# #GRADIENT DESCENT
# #insert dummy feature in matrix
# new_col=np.ones(N)
# X1_train = np.insert(X_train, 0, new_col, axis=1) # dummy feature was included
# alpha = 1 #learning rate
# #initialize w
# w = np.array([-10,-10,-10]) #initial vetor of weights
# z = np.zeros(N)
# IT = 300
# gr_norms = np.zeros(IT) # to store squared norm of gradient at ecah iteration
# cost = np.zeros(IT)  # to store the cost at each iteration
# for n in range(IT):
#     z = np.dot(X1_train,w)
#     #y = 1/np.logaddexp(0, -z)
#     y = 1/(1 + np.exp(-z))
#     diff = y-tt_train
#     gr = np.dot(X1_train.T, np.transpose(diff.T))/N # this is the gradient
#     #compute squared norm of the gradient
#     gr_norm_sq = np.dot(gr,gr)
#     gr_norms[n] = gr_norm_sq
#     #update the vector of parameters
#     w = w - alpha * gr 
#     #compute the cost
#     cost[n] = 0
#     for i in range(N):
#         cost[n] += tt_train[i]*np.logaddexp(0, -z[i]) + (1-tt_train[i])*np.logaddexp(0,z[i])
#     cost[n] = cost[n]/N
# print(w)
# print(cost[:5])
# print(cost[IT-5:IT])
# print(gr_norms[IT-5:IT])

# # plot change of cost during gradient descent
# lin = np.linspace(1,IT,IT)
# plt.scatter(lin, cost, color = 'blue')
# plt.title('Change in Cost function')
# plt.xlabel('cost')
# plt.ylabel('iteration number')
# #plt.scatter(lin, gr_norms, color = 'red')
# plt.show()

# #plot classes
# v1 = np.linspace(4,6,100)
# print(w)
# v2 = -w[0]/w[2]-w[1]/w[2]*v1 # this is the same as the equation w0 + w1v1 + w2v2 =0
# plt.scatter(X_train_0[:,0], X_train_0[:,1], color = 'red')
# plt.scatter(X_train_1[:,0], X_train_1[:,1], color = 'blue')
# plt.scatter(X_train_2[:,0], X_train_2[:,1], color = 'green')
# plt.plot(v1,v2, color = 'orange') #decision boundary
# plt.show()

# #compute test error
# new_col=np.ones(M)
# X1_test = np.insert(X_test, 0, new_col, axis=1) # dummy feature was included
# z = np.dot(X1_test,w)
# y = np.zeros(M)
# for i in range(M):
#     if(z[i]>=0):
#         y[i]=1
# u = y - tt_test
# print(u)
# err = np.count_nonzero(u)/M  #mislassification rate
# print(err)

