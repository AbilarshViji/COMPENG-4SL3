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

def determineBestFeatureAndError(xTrain, xTest, tTrain, tTest): #Determine best features for each k, and compute crossvalidation and testing errors
    trainingSet = np.ones(np.size(xTrain.T, 0)) #Initalize training dummy
    testingSet = np.ones(np.size(xTest.T, 0)) #Initialize testing dummy
    rankings = []
    crossValidationError = []
    testingError = []
    # Determine best feature, calculate KFold and testing errors
    featuresRemaining = list(range(len(xTrain)))
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
    return rankings, crossValidationError, testingError

figCount = 0
def plotModel(rankings, crossValidationError, testingError, title): #Plots crossvalidation and testing error
    global figCount #Keeps track of figure count
    plt.figure(figCount, figsize=(9, 6))
    figCount += 1
    k = skds.load_boston().feature_names[rankings] #X axis labels 
    plt.plot(k, crossValidationError, "-o")
    dataLabels(plt, k, crossValidationError)
    plt.plot(k, testingError, "-o")
    dataLabels(plt, k, testingError)
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend(["Cross-Validation Error", "Testing Error"])
    plt.savefig(title+".png", dpi=120)

def dataLabels(plot, x, y):
    for xVal, yVal in zip(x, y):
        plot.annotate(np.round(yVal, 3), (xVal,yVal))


#import data set from scikit
rand = 1727
x, target = skds.load_boston(return_X_y=True)

#Splits training and testing set
xTrain, xTest, tTrain, tTest = skms.train_test_split(x, target, test_size = 1/4, random_state = rand)
xTrain = xTrain.T
xTest = xTest.T

ranking, crossValidationError, testingError = determineBestFeatureAndError(xTrain, xTest, tTrain, tTest) #compute ranking, crossvalidation error and testing error for all features
plotModel(ranking, crossValidationError, testingError, "Errors vs Size of Subset of Features (k)") #Plot errors

#Initialize basis function dataset
logTrain = np.log(xTrain[0])
logTest = np.log(xTest[0])
sqTrain = np.power(xTrain, 2)
sqTest = np.power(xTest, 2)
sqrtTrain = np.sqrt(xTrain)
sqrtTest = np.sqrt(xTest)

for k in range(1, len(xTrain)):  #Apply basis function (log) to all features
    if 0 in xTrain[k] or 0 in xTest[k]: #Log(0) = inf, so if 0 is in feature, do not apply log
        logTrain = np.vstack((logTrain, xTrain[k]))
        logTest = np.vstack((logTest, xTest[k]))
    else: #else apply log function to feature
        logTrain = np.vstack((logTrain, np.log(xTrain[k])))
        logTest = np.vstack((logTest, np.log(xTest[k])))

basisRanking, crossValidationBasisError, testingBasisError = determineBestFeatureAndError(logTrain, logTest, tTrain, tTest) #compute ranking, crossvalidation error and testing error for all features with log basis function applied
plotModel(basisRanking, crossValidationBasisError, testingBasisError, "Basis (log) Errors vs Size of Subset of Features (k)") #Plot errors

basisRanking, crossValidationBasisError, testingBasisError = determineBestFeatureAndError(sqTrain, sqTest, tTrain, tTest) #compute ranking, crossvalidation error and testing error for all features with sq basis function applied
plotModel(basisRanking, crossValidationBasisError, testingBasisError, "Basis (square) Errors vs Size of Subset of Features (k)") #Plot errors

basisRanking, crossValidationBasisError, testingBasisError = determineBestFeatureAndError(sqrtTrain, sqrtTest, tTrain, tTest) #compute ranking, crossvalidation error and testing error for all features with sqrt basis function applied
plotModel(basisRanking, crossValidationBasisError, testingBasisError, "Basis (square root) Errors vs Size of Subset of Features (k)") #Plot errors

plt.show()