import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

def initLayers(inputSize, hidden1, hidden2):
    layers = [[inputSize, hidden1], [hidden1, hidden2], [hidden2, 1]]
    params = dict()
    for layerNum, [inp, out] in enumerate(layers, 1):
        params['W' + str(layerNum)] = np.random.randn(out, inp)*0.01
        print( params['W' + str(layerNum)].shape)
        params['b' + str(layerNum)] = np.zeros((out, 1))
        # params['b' + str(layerNum)] = np.random.randn(out, 1)*0.01
    return layers, params

def relu(Z):
    # return 1/(1+np.exp(-Z))  #TODO LOOK HHERE RELU
    return np.maximum(0, Z)

# def reluPrime(Z):
#     # sig = sigmoid(Z)
#     # return sig*(1-sig)  #dA*sig*(1-sig)
#     return np.array([[0 if x<0 else 1 for x in Z[0]]])

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# def sigmoidPrime(dA, Z):
#     sig = sigmoid(Z)
#     return sig*(1-sig)  #dA*sig*(1-sig)

def forwardPropagation(X, params, layers):
    # backward = {}
    # ACurr = X

    # for layerNum in range(1, len(layers)+1):
    #     APrev = ACurr
    #     # WCurr = np.dot(params['W' + str(layerNum)], APrev) + params['b' + str(layerNum)]
    #     ZCurr = np.dot(params['W' + str(layerNum)], APrev) + params['b' + str(layerNum)]
    #     # ZCurr = np.dot(WCurr, APrev)
    #     ACurr = relu(ZCurr)
    #     # if layerNum == len(layers)-1:
    #     #     ACurr = sigmoid(ZCurr) 
    #     backward['A' + str(layerNum)] = ACurr
    #     backward['Z' + str(layerNum)] = ZCurr
    #     # print(APrev.shape, ZCurr.shape)

    backward = {}
    
    backward['Z1'] = np.dot(params['W1'], X)+params['b1']
    backward['A1'] = relu(backward['Z1'])

    backward['Z2'] = np.dot(params['W2'], backward['A1'])+params['b2']
    backward['A2'] = relu(backward['Z2'])

    backward['Z3'] = np.dot(params['W3'], backward['A2'])+params['b3']
    backward['A3'] = sigmoid(backward['Z3'])
    return backward['A3'], backward
    return APrev, backward

def backwardPropagation(y, backward, params, layers, X):
    # gradientValues = {}
    # #layer 3
    # m = X.shape[1]
    # gradz3J = backward['A3'] - y
    # gradientValues['dW3'] = np.dot(gradz3J, backward['A2'].T)/m
    # gradientValues['db3'] = np.sum(gradz3J, axis=1, keepdims=True)/m

    # #layer2
    # gradz2J = np.dot(params['W3'].T, gradz3J) * (1-np.power(backward['A2'], 2))
    # gradientValues['dW2'] = np.dot(gradz3J, gradz2J.T)/m
    # gradientValues['db2'] = np.sum(gradz2J, axis=1, keepdims=True)/m

    # #layer1
    # gradz1J = np.dot(params['W2'].T, gradz2J) * (1-np.power(backward['A1'], 2))
    # gradientValues['dW1'] = np.dot(gradz1J, X.T)/m
    # gradientValues['db1'] = np.sum(gradz1J, axis=1, keepdims=True)/m


    gradientValues = {}
    #layer 3
    m = X.shape[1]
    gradz3J = backward['A3'] - y
    gradientValues['dW3'] = np.dot(gradz3J, backward['A2'].T)/m
    gradientValues['db3'] = np.sum(gradz3J, axis=1, keepdims=True)/m

    #layer2
    gradz2J = np.dot(params['W3'].T, gradz3J) * (1-np.power(backward['A2'], 2))
    gradientValues['dW2'] = np.dot(gradz3J, gradz2J.T)/m
    gradientValues['db2'] = np.sum(gradz2J, axis=1, keepdims=True)/m

    #layer1
    gradz1J = np.dot(params['W2'].T, gradz2J) * (1-np.power(backward['A1'], 2))
    gradientValues['dW1'] = np.dot(gradz1J, X.T)/m
    gradientValues['db1'] = np.sum(gradz1J, axis=1, keepdims=True)/m
    return gradientValues


def computeCost(yPred, y):
    m = yPred.shape[1]
    cost = -1/m * np.sum((np.multiply(y, np.log(yPred).T) + np.multiply(1-y, np.log(1-yPred).T)))
    return cost

def computeError(yPred, y):
    yPred = np.around(yPred)
    # print(yPred)
    return 1-((np.dot(y,yPred.T) + np.dot(1-y,1-yPred.T))/float(y.size))

def update(params, gradientValues, layers):
    LR = 1.2
    for layerNum in range(1, len(layers)+1):
        print(params['W' + str(layerNum)], params['W' + str(layerNum)].shape)
        print("DW", gradientValues['dW' + str(layerNum)], gradientValues['dW' + str(layerNum)].shape)
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

for numFeatures in [2]:
    for n1 in [2]:
        for n2 in [3]:
# for numFeatures in range(2, 5):
#     for n1 in range(2, 5):
#         for n2 in range(2, 5):
            bestCostsTrain = [0]
            bestErrorsTrain = [1]
            bestCostsVal = [0]
            bestErrorsVal = [1]
            for diffWeightAssignments in range(15):
                costsTrain = []
                errorsTrain = []
                costsVal = []
                errorsVal = []
                layers, params = initLayers(numFeatures, n1, n2)
                for i in range(100):
                    xTrainShuffle, tTrainShuffle = shuffle(xTrain, tTrain)
                    # xTrainShuffle, tTrainShuffle = xTrain, tTrain
                    yPred, backward = forwardPropagation(xTrainShuffle.T[:numFeatures], params, layers)
                    cost = computeCost(yPred, tTrainShuffle)
                    costsTrain.append(cost)
                    error = computeError(yPred, tTrainShuffle)
                    errorsTrain.append(error[0])
                    
                    gradientValues = backwardPropagation(tTrainShuffle, backward, params, layers, xTrainShuffle.T[:numFeatures])
                    print(params.keys(), gradientValues.keys())
                    params = update(params, gradientValues, layers)
                    yPred, _ = forwardPropagation(xVal.T[:numFeatures], params, layers)
                    costsVal.append(computeCost(yPred, tVal))

                    errorsVal.append(computeError(yPred, tVal)[0])
                # print(errorsVal)
                if (bestErrorsVal[-1]) > (errorsVal[-1]):
                    bestErrorsVal = errorsVal
                    bestCostsTrain = costsTrain
                    bestErrorsTrain = errorsTrain
                    bestCostsVal = costsVal
            plot(bestCostsTrain, bestCostsVal, bestErrorsTrain, bestErrorsVal, numFeatures, n1, n2)
            print(numFeatures, n1, n2)
