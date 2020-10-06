import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.model_selection as skms
import sklearn.metrics as skmt
import sklearn.linear_model as sklm
import sklearn.metrics
#https://cdn1.sph.harvard.edu/wp-content/uploads/sites/565/2018/08/572Spr07-BasisMethods-4up.pdf
#https://chrispfchung.github.io/model%20data/boston-housing-data/#linear-relationships

def computePrediction(x, w): #Computes prediction on a given set of x points
    # pred = []
    # # M = len(w) - 1
    # pred = []
    # for feature in x:
    #     curr = 0
    #     for m, value in enumerate(feature):
    #         curr += w[m] * value
    #     pred.append(curr)
    # return pred
    print(x, w)
    return np.dot(x, w)


def computeWeights(trainingData, targetData):
    # print(trainingData)
    A = np.dot(np.transpose(trainingData), trainingData)
    c = np.dot(np.transpose(trainingData), targetData)
    w = np.dot(np.linalg.inv(A), c)
    # print('a', A)
    # print('c', c)
    # print(trainingData.shape, targetData.shape, A, c, w, sep="\n")
    return w

def calcError(predictedList, actualList): #Computes least squares error
    # error = 0
    # for predict, actual in zip(predictedList, actualList): #Iterate through both predicted and actual list
    #     error += (predict-actual)**2 #Sum difference squared
    #     # print(predict-actual)
    # return error/len(predictedList)
    diff = np.subtract(actualList, predictedList)
    return np.dot(diff, np.transpose(diff))/len(diff)

#import data set from scikit
rand = 1727
x, target = skds.load_boston(return_X_y=True)
xIterate = x.transpose()
errors = []
# print(target)
# # initialize error values
# for featureIndex, feature in enumerate(xIterate):
#     currentFoldError = []
#     for train, valid in skms.KFold(5, shuffle=True, random_state=rand).split(feature):
#         training = feature[train]
#         targetTrain = target[train]
#         w = computeWeights(training, targetTrain)
        
#         validation = feature[valid]
#         targetValid = target[valid]
#         validationOutput = computePrediction(validation, w)
#         currentFoldError.append(skmt.r2_score(targetValid, validation))
#         print(currentFoldError)
#         # currentFoldError.append(calcError(validationOutput, targetValid))
#     errors.append((featureIndex, np.average(currentFoldError)))
# errors.sort(key=lambda x:x[1], reverse=True)
# print('errors', errors)

rankings = []
for featureIndex, feature in enumerate(xIterate):
    rankings.append((featureIndex, np.corrcoef(target, feature)[0,1]))
rankings.sort(key=lambda x:abs(x[1]), reverse=True)

# print(rankings)
# print(target)
trainingSet = np.ones(506)
KFoldError = []
for index, error in rankings:
    trainingSet = np.vstack((trainingSet, xIterate[index]))
    w = computeWeights(np.transpose(trainingSet), target)
    # print(w)
    # w[0] = 0
    # print(np.transpose(trainingSet)[0])
    validationOutput = computePrediction(np.transpose(trainingSet), w) # TODO: VALIDATION SET
    # print(validationOutput[:8])
    KFoldError.append(np.sqrt(calcError(validationOutput, target)))

    # trainingSet = np.vstack((trainingSet, xIterate[index]))
    reg = sklm.LinearRegression().fit(np.transpose(trainingSet), target)
    # print(reg.score(np.transpose(trainingSet), target))
    # KFoldError.append((sklearn.metrics.mean_squared_error(target, reg.predict(np.transpose(trainingSet))))) #RMSE
    # print(reg.coef_)
    # x = input()
print("hi",KFoldError)
# print(target)
# print(xIterate[3])