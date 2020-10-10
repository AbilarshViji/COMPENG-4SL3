import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.model_selection as skms

def computePrediction(x, w): #Computes prediction on a given set of x points
    return np.dot(x, w)

def computeWeights(trainingData, targetData):
    # print(trainingData)
    A = np.dot(trainingData.T, trainingData)
    c = np.dot(trainingData.T, targetData)
    w = np.dot(np.linalg.inv(A), c)
    return w

def calcError(predictedList, actualList): #Computes least squares error
    diff = np.subtract(actualList, predictedList)
    return np.dot(diff, diff.T)/len(diff)

def average(listToAverage):
    return sum(listToAverage)/len(listToAverage)

def trainAndGetError(training, valid, target, validTarget):
    w = computeWeights(training, target)
    validationOutput = computePrediction(valid, w)
    return calcError(validationOutput, validTarget)

#import data set from scikit
rand = 1727
x, target = skds.load_boston(return_X_y=True)
xIterate = x.T

# KFoldError = []
KFoldBasisError = []
testError = []
testBasisError = []
featureNamesOrder = ""

xTrain, xTest, tTrain, tTest = skms.train_test_split(x, target, test_size = 1/4, random_state = rand)
trainingSet = np.ones(np.size(xTrain,0))
testingSet = np.ones(np.size(xTest,0))
xTrain = xTrain.T
xTest = xTest.T

rankings = []
featuresRemaining = list(range(0, 13))
# for k in range(13, 0, -1):
crossValidationError = []
crossValidationBasisError = []
testingError = []
testingBasisError = []
  
#Crossfold non basis
while featuresRemaining:
    foldError = []
    for feature in featuresRemaining:
        currentError = []
        for train, valid in skms.KFold(5, shuffle=True, random_state=rand).split(trainingSet.T):
            current = np.vstack((trainingSet, xTrain[feature]))
            currentError.append(trainAndGetError(current.T[train], current.T[valid], tTrain[train], tTrain[valid]))
        foldError.append((feature, average(currentError)))
    
    foldError.sort(key=lambda x:x[1])
    bestFeature = foldError[0][0]
    crossValidationError.append(foldError[0][1])
    trainingSet = np.vstack((trainingSet, xTrain[bestFeature]))
    
    testingSet = np.vstack((testingSet, xTest[bestFeature]))
    testingError.append(trainAndGetError(trainingSet.T, testingSet.T, tTrain, tTest))
    
    rankings.append(bestFeature)
    featuresRemaining.remove(bestFeature)

print("RANMFLKNALNLFNS", rankings, crossValidationError, testingError)
# trainingSet = np.ones(np.size(xTrain.T,0))
# testingSet = np.ones(np.size(xTest.T,0))

for k in range(2,len(trainingSet)):
    logTrain = np.vstack((trainingSet[0], (np.sqrt(trainingSet[1:k]))))
    logTest = np.vstack((testingSet[0], (np.sqrt(testingSet[1:k]))))
    # print(k, logTrain)
    for train, valid in skms.KFold(5, shuffle=True, random_state=rand).split(logTrain.T):
        currentError.append(trainAndGetError(logTrain.T[train], logTrain.T[valid], tTrain[train], tTrain[valid]))
    crossValidationBasisError.append(average(currentError))
    testingBasisError.append(trainAndGetError(logTrain.T, logTest.T, tTrain, tTest))

print(crossValidationBasisError)
print(testingBasisError)
# for index, error in enumerate(rankings):

#     trainingSet = np.vstack((trainingSet, xTrain[index]))
#     testingSet = np.vstack((testingSet, xTest[index]))
#     currentCVError = []
#     currentCVBasisError = []
#     currentTestError = []
#     currentTestBasisError = []

#     featureNamesOrder += skds.load_boston().feature_names[index] + ","

#     for train, valid in skms.KFold(5, shuffle=True, random_state=rand).split(trainingSet.T):

#         # compute standard model
#         w = computeWeights(trainingSet.T[train], tTrain[train])
#         validationOutput = computePrediction(trainingSet.T[valid], w)
#         currentCVError.append(calcError(validationOutput, tTrain[valid]))

#         testOutput = computePrediction(testingSet.T, w)
#         currentTestError.append(calcError(testOutput, tTest))

#         #compute basis expansion model (ln)
#         logTrainingSet = np.copy(trainingSet)
#         for val in range(len(logTrainingSet[1])):
#             logTrainingSet[1][val] = np.log(logTrainingSet[1][val])            
#         w = computeWeights(logTrainingSet.T[train], tTrain[train])
#         validationBasisOutput = computePrediction(logTrainingSet.T[valid], w)
#         currentCVBasisError.append(calcError(validationBasisOutput, tTrain[valid]))

#         logTestingSet = np.copy(testingSet)
#         for val in range(len(logTestingSet[1])):
#             logTestingSet[1][val] = np.log(logTestingSet[1][val])
#         testBasisOutput = computePrediction(logTestingSet.T, w)
#         currentTestBasisError.append(calcError(testBasisOutput, tTest))
        
#     KFoldError.append(average(currentCVError))
#     KFoldBasisError.append(average(currentCVBasisError))
#     testError.append(average(currentTestError))
#     testBasisError.append(average(currentTestBasisError))

# print(KFoldError)
# print(KFoldBasisError)
# print(testError)
# print(testBasisError)
# x = list(range(1, 1))
# plt.figure(0)
# plt.plot(x, KFoldError)
# plt.plot(x, KFoldBasisError)
# plt.plot(x, testError)
# plt.plot(x, testBasisError)
# plt.xlabel("k")
# plt.ylabel("Error")
# plt.title("Errors vs Size of Subset of Features (k)")
# plt.legend(["Cross-Validation Error", "Cross-Validation Basis Error", "Test Error", "Test Basis Error"])
# plt.show()
