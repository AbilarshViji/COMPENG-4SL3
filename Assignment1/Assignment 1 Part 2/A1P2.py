import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.model_selection as skms

def computePrediction(x, w): #Computes prediction on a given set of x points
    return np.dot(x, w)

def computeWeights(trainingData, targetData): #Computes weights
    A = np.dot(trainingData.T, trainingData)
    c = np.dot(trainingData.T, targetData)
    w = np.dot(np.linalg.inv(A), c)
    return w

def calcError(predictedList, actualList): #Computes least squares error
    diff = np.subtract(actualList, predictedList)
    return np.dot(diff, diff.T)/len(diff)

def average(listToAverage): #Computes average
    return sum(listToAverage)/len(listToAverage)

def trainAndGetError(training, valid, target, validTarget): #Computes weight, prediction, and error
    w = computeWeights(training, target)
    validationOutput = computePrediction(valid, w)
    return calcError(validationOutput, validTarget)

#import data set from scikit
rand = 1727
x, target = skds.load_boston(return_X_y=True)

#Splits training and testing set
xTrain, xTest, tTrain, tTest = skms.train_test_split(x, target, test_size = 1/4, random_state = rand)
trainingSet = np.ones(np.size(xTrain,0)) #Initalize training dummy
testingSet = np.ones(np.size(xTest,0)) #Initialize testing dummy
xTrain = xTrain.T
xTest = xTest.T

rankings = [] 
featuresRemaining = list(range(0, 13))
crossValidationError = []
crossValidationBasisError = []
testingError = []
testingBasisError = []
  
# Determine best feature, calculate KFold and testing errors
while featuresRemaining: #Runs while features are remaining
    foldError = []
    for feature in featuresRemaining: #Iterate over each remaining feature
        currentError = []
        for train, valid in skms.KFold(5, shuffle=True, random_state=rand).split(trainingSet.T): #Iterate over 5 KFoldes
            current = np.vstack((trainingSet, xTrain[feature]))
            currentError.append(trainAndGetError(current.T[train], current.T[valid], tTrain[train], tTrain[valid]))
        foldError.append((feature, average(currentError))) #Append error for current feature
    
    foldError.sort(key=lambda x:x[1]) #Sort list by error
    bestFeature = foldError[0][0] #Save best feature
    crossValidationError.append(foldError[0][1]) #Save best error
    trainingSet = np.vstack((trainingSet, xTrain[bestFeature])) #Add best feature to training test
    
    testingSet = np.vstack((testingSet, xTest[bestFeature])) #Add best feature to testing set
    testingError.append(trainAndGetError(trainingSet.T, testingSet.T, tTrain, tTest)) #Compute testing error
    
    rankings.append(bestFeature) #Save best feature
    featuresRemaining.remove(bestFeature) #Remove best feature from features remaining

# Compute basis KFold and testing error
logTrain = trainingSet[0]
logTest = testingSet[0]

for k in range(1,len(trainingSet)):
    if 0 in trainingSet[k] or 0 in testingSet[k]: #Log(0) = inf, so if 0 is in feature, do not apply log
        logTrain = np.vstack((logTrain, trainingSet[k]))
        logTest = np.vstack((logTest, testingSet[k]))
    else: #else apply log function to feature
        logTrain = np.vstack((logTrain, np.log(trainingSet[k])))
        logTest = np.vstack((logTest, np.log(testingSet[k])))
    for train, valid in skms.KFold(5, shuffle=True, random_state=rand).split(logTrain.T): #Iterate over 5 KFolds
        currentError.append(trainAndGetError(logTrain.T[train], logTrain.T[valid], tTrain[train], tTrain[valid])) #Get errors for current feature
    crossValidationBasisError.append(average(currentError)) #Append average error (cross validation error for basis function)
    testingBasisError.append(trainAndGetError(logTrain.T, logTest.T, tTrain, tTest)) #Compute and save testing error

k = list(range(1, 14)) #Initialize k for graphing
plt.figure(0)
plt.plot(k, crossValidationError)
plt.plot(k, crossValidationBasisError)
plt.plot(k, testingError)
plt.plot(k, testingBasisError)
plt.xlabel("k")
plt.ylabel("Error")
plt.title("Errors vs Size of Subset of Features (k)")
plt.legend(["Cross-Validation Error", "Cross-Validation Basis Error", "Test Error", "Test Basis Error"])
plt.show()