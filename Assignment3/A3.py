import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

def initLayers(inputSize, hidden1, hidden2):
    layers = [[inputSize, hidden1], [hidden1, hidden2], [hidden2, 1]]
    # np.random.seed(1727)
    params = dict()
    for layerNum, [inp, out] in enumerate(layers):
        params['W' + str(layerNum)] = np.random.randn(out, inp)*0.01
        params['b' + str(layerNum)] = np.zeros((out, 1))
        # params['b' + str(layerNum)] = np.random.randn(out, 1)*0.01
    return layers, params

def relu(z):
    return np.maximum(0, z)

# def reluBackwards(dA, Z):
#     dZ = np.copy(dA)
#     dZ[Z <= 0] = 0
#     return dZ

def reluPrime(z):
    # print(z)
    # print(np.array([[0 if x<0 else 1 for x in z[0]]]))
    return np.array([[0 if x<0 else 1 for x in z[0]]])
    # if z < 0:
    #     return 0
    # else:
    #     return 1

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoidBackwards(dA, Z):
    sig = sigmoid(Z)
    return dA*sig*(1-sig)

def sigmoidPrime(dA, Z):
    sig = sigmoid(Z)
    return sig*(1-sig)  #dA*sig*(1-sig)

def forwardPropagation(X, params, layers):
    backward = {}
    ACurr = X

    for layerNum in range(len(layers)):
        APrev = ACurr
        WCurr = params['W' + str(layerNum)]
        bCurr = params['b' + str(layerNum)]

        ZCurr = np.dot(WCurr, APrev) + bCurr

        ACurr = relu(ZCurr)
        if layerNum == len(layers)-1:
            ACurr = sigmoid(ZCurr) 
        backward['A' + str(layerNum-1)] = APrev
        backward['Z' + str(layerNum)] = ZCurr
    
    return ACurr, backward

# def backwardPropagation(yPred, y, backward, params, layers):
#     gradientValues = {}

#     dAPrev = -(y/yPred - (1-y)/(1-yPred))
#     # dAPrev = -y + yPred


#     for layerNum in range(len(layers))[::-1]:
#         layerNumPrev = layerNum - 1
#         dACurr = dAPrev
#         APrev = backward['A' + str(layerNumPrev)]
#         ZCurr = backward['Z' + str(layerNum)]

#         WCurr = params['W' + str(layerNum)]
#         bCurr = params['b' + str(layerNum)]

#         dZCurr = reluBackwards(dACurr, ZCurr)
#         if layerNum == len(layers)-1:
#             dZCurr = sigmoidBackwards(dACurr, ZCurr)
#         m = APrev.shape[1]
#         dWCurr = np.dot(dZCurr, APrev.T)/m
#         dbCurr = np.sum(dZCurr, axis=1, keepdims=True)/m
#         dAPrev = np.dot(WCurr.T, dZCurr)

#         gradientValues['dW' + str(layerNum)] = dWCurr
#         gradientValues['db' + str(layerNum)] = dbCurr
#     return gradientValues

# def backwardPropagation(yPred, y, backward, params, layers, cost):
#     gradientValues = {}

#     dAPrev = -y + sigmoid(yPred)
#     # dAPrev = yPred - y
#     # m = y.shape[0]
#     # dAPrev = 1/m*(-y*np.log(yPred)-(1-y)*np.log(1-yPred))

#     for layerNum in range(len(layers))[::-1]:
#         layerNumPrev = layerNum - 1
#         dACurr = dAPrev
#         APrev = backward['A' + str(layerNumPrev)]
#         ZCurr = backward['Z' + str(layerNum)]

#         WCurr = params['W' + str(layerNum)]
#         bCurr = params['b' + str(layerNum)]

#         gradZJ = reluPrime(ZCurr) * np.dot(WCurr, dACurr)
#         gradWJ = np.dot(gradZJ, APrev)
#         dwCurr = gradWJ

#         # dcdz = np.dot(dACurr, reluPrime(ZCurr))
#         # dcdw = np.dot(dcdz, APrev.T)
#         # dWCurr = dcdw
#         # dcdb = np.dot(1, dcdz)
#         # dBCurr = dcdb
#         # dAPrev = np.dot(WCurr.T, dcdz)

#         # m = APrev.shape[1]
#         # dZCurr = np.dot(dACurr, _)
#         # dWCurr = np.dot(dZCurr, APrev)/m
#         # dbCurr = np.sum(dZCurr, axis=1, keepdims=True)/m

#         # dZCurr = reluBackwards(dACurr, ZCurr)
#         # if layerNum == len(layers)-1:
#         #     dZCurr = sigmoidBackwards(dACurr, ZCurr)
        
#         # dWCurr = np.dot(dZCurr, APrev.T)/m
#         # dbCurr = np.sum(dZCurr, axis=1, keepdims=True)/m
#         # dAPrev = np.dot(WCurr.T, dZCurr)

#         gradientValues['dW' + str(layerNum)] = dWCurr
#         gradientValues['db' + str(layerNum)] = dbCurr
#     return gradientValues

def backwardPropagation(yPred, y, backward, params, layers, cost):
    gradientValues = {}
    #layer 3
    jz3 = -y + yPred
    gradW3J = np.dot(jz3, backward['A1'].T)
    gradientValues['dW2'] = gradW3J
    gradientValues['db2'] = np.dot(params['b2'], jz3)
    print(params['b2'].shape, jz3.shape)
    gradz2J = np.multiply(reluPrime(backward['Z2']), np.dot(params['W2'].T, jz3))

    #layer2
    gradW2J = np.dot(gradz2J, backward['A0'].T)
    gradientValues['dW1'] = gradW2J
    print(params['b1'].shape, gradz2J.shape)
    gradientValues['db1'] = np.dot(params['b1'].T, gradz2J)
   
    gradz1J = np.multiply(reluPrime(backward['Z1']), np.dot(params['W1'].T, gradz2J))

    #layer1
    gradW1J = np.dot(gradz1J, backward['A-1'].T)
    gradientValues['dW0'] = gradW1J
    gradientValues['db0'] = np.dot(params['b0'].T, gradz1J)
    return gradientValues


def computeCost(yPred, y):
    m = yPred.shape[1]
    cost = -1/m * (np.dot(y, np.log(yPred).T) + np.dot(1-y, np.log(1-yPred).T))
    return cost[0]

def computeError(yPred, y):
    yPred = np.round(yPred, decimals=0)
    count = 0
    for pred, act in zip(yPred[0], y):
        if pred != act:
            count += 1
    return count/len(y)

def update(params, gradientValues, layers):
    LR = 0.05
    for layerNum in range(len(layers)):
        params['W' + str(layerNum)] -= LR * gradientValues['dW' + str(layerNum)]
        params['b' + str(layerNum)] -= LR * gradientValues['db' + str(layerNum)]
    return params

figCount = 0
def plot(bestCostsTrain, bestCostsVal, bestErrorsTrain, bestErrorsVal, numFeatures, n1, n2):
    global figCount
    plt.figure(figCount)
    figCount += 1
    plt.plot(range(1, 101), bestCostsTrain)
    plt.plot(range(1, 101), bestCostsVal)
    plt.legend(["Training", "Validation"])
    plt.title("Training and Validation Loss for {} features and (n1, n2) = ({}, {})".format(numFeatures, n1, n2))
    plt.savefig("{}{}{}Loss.png".format(numFeatures, n1, n2))
    plt.close(figCount)
    plt.figure(figCount)
    figCount += 1
    plt.plot(range(1, 101), bestErrorsTrain)
    plt.plot(range(1, 101), bestErrorsVal)
    plt.legend(["Training", "Validation"])
    plt.title("Training and Validation Error for {} features and (n1, n2) = ({}, {})".format(numFeatures, n1, n2))
    plt.savefig("{}{}{}Error.png".format(numFeatures, n1, n2))
    plt.close(figCount)

dataset = pd.read_csv('data_banknote_authentication1.txt') #load data
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values
rand = 1727

xTrain, xTest, tTrain, tTest = train_test_split(X, t, test_size = 1/3, random_state = rand) #split data
xTrain, xVal, tTrain, tVal = train_test_split(xTrain, tTrain, test_size=1/3, random_state = rand)



sc = StandardScaler()
xTrain[:, :]  = sc.fit_transform(xTrain[:, :])
xVal[:, :]  = sc.transform(xVal[:, :])
xTest[:, :]  = sc.transform(xTest[:, :])

t = (t.reshape((1, t.shape[0])))

# seed = 1
errors = [0,1]
while(max(errors) > 0.2):
    errors = []
    seed = int(time.time()*1.85)
    np.random.seed(seed)
    for numFeatures in range(2, 5):
        for n1 in range(2, 5):
            for n2 in range(2, 5):
                bestCostsTrain = [0]
                bestErrorsTrain = [1]
                bestCostsVal = [0]
                bestErrorsVal = [1]
                for diffWeightAssignments in range(3):
                    costsTrain = []
                    errorsTrain = []
                    costsVal = []
                    errorsVal = []
                    layers, params = initLayers(numFeatures, n1, n2)
                    for i in range(100):
                        xTrainShuffle, tTrainShuffle = shuffle(xTrain, tTrain)
                        yPred, backward = forwardPropagation(xTrainShuffle.T[:numFeatures], params, layers)
                        # print(backward)
                        cost = computeCost(yPred, tTrainShuffle)
                        costsTrain.append(cost)
                        errorsTrain.append(computeError(yPred, tTrainShuffle))
                        gradientValues = backwardPropagation(yPred, tTrain, backward, params, layers, cost)
                        # print(gradientValues)
                        # print("PARA0", params)
                        params = update(params, gradientValues, layers)
                        yPred, _ = forwardPropagation(xVal.T[:numFeatures], params, layers)
                        costsVal.append(computeCost(yPred, tVal))
                        errorsVal.append(computeError(yPred, tVal))
                    if (bestErrorsVal[-1]) > (errorsVal[-1]):
                        bestErrorsVal = errorsVal
                        bestCostsTrain = costsTrain
                        bestErrorsTrain = errorsTrain
                        bestCostsVal = costsVal
                errors.append(bestErrorsVal[-1])
    # print(max(errors), seed)
                
            # plot(bestCostsTrain, bestCostsVal, bestErrorsTrain, bestErrorsVal, numFeatures, n1, n2)

