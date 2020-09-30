import numpy as np
import matplotlib.pyplot as plt

def computePrediction(x, w, M): #Computes prediction on a given set of x points
    pred = []
    for val in x: #Iterate through each value in list x
        curr = 0
        for m in range(M+1):
            curr += w[m]*(val**m) #Sum predicted value
        pred.append(curr) #Append prediction to list
    return pred

figCount = 0
def plotModel(model, x, M, w, points, title): #Plots data with true function, and predicted function
    global figCount #Keeps track of figure count
    plt.figure(figCount)
    figCount += 1
    plt.scatter(x, points, color="green") #Plot points
    plt.title(title+ " M = " + str(M))
    x = np.linspace(0.,1.,100)
    poly = computePrediction(x, w, M) #Compute values for polynomial function based on w
    plt.plot(x, poly, color="cyan", linewidth=2) #Plot polynomial function
    plt.plot(x, np.sin(4*np.pi*x), color="magenta", linewidth=2) #Plot actual function
    plt.legend(["Predicted Function", "Actual Function", "Actual Points"])
    plt.savefig(title+str(M)+".png")

def calcError(predictedList, actualList): #Computes least squares error
    error = 0
    for predict, actual in zip(predictedList, actualList): #Iterate through both predicted and actual list
        error += (predict-actual)**2 #Sum difference squared
    return error/len(predictedList)

def computeTrainingData(maxM, xTrain): #Computes training data
    trainingData = np.ones(len(xTrain)) #Initialize training data with dummy
    for m in range(1, maxM + 1): #Iterate through remaining M values
        row = []
        for x in xTrain:
            row.append(x**m) #Append x**m to row
        trainingData = np.vstack((trainingData, row)) #Add row to training data
    trainingData = np.transpose(trainingData) #Transpose final matrix
    return trainingData

#N = 10, num training samples
#D = 0-9, num features
xTrain = np.linspace(0.,1.,10) #training set
xValid = np.linspace(0.,1.,100) #validation set
np.random.seed(1727)
tTrain = np.sin(4*np.pi*xTrain) + 0.3 * np.random.randn(10)
tValid = np.sin(4*np.pi*xValid) + 0.3 * np.random.randn(100) 

maxM = 9

trainingError = []
validationError = []

for m in range(maxM+1): #Iterate over all M values
    trainingData = computeTrainingData(m, xTrain) #Generate training data
    A = np.dot(np.transpose(trainingData), trainingData)
    c = np.dot(np.transpose(trainingData), tTrain)
    #Compute w weights
    if m == 0: #Edge case for m=0 since np.linalg.inv does not support 1D array
        w = [np.dot(1/A, c)]
    else:    
        w = np.dot(np.linalg.inv(A), c)
    
    trainingOutput = computePrediction(xTrain, w, m) #Compute training prediction
    plotModel(trainingOutput, xTrain, m, w, tTrain, "Training Data") #Plot training data
    trainingError.append(calcError(trainingOutput, tTrain)) #Append training error

    validationOutput = computePrediction(xValid, w, m) #Compute validation prediction
    plotModel(validationOutput, xValid, m, w, tValid, "Validation Data") #Plot validation data
    validationError.append(calcError(validationOutput, tValid)) #Append validation error

#Plot training and validation error
plt.figure(figCount)
figCount += 1
plt.plot(trainingError,marker='o', linewidth=2)
plt.plot(validationError, marker='o', linewidth=2)
plt.title("Training and Validation Error")
plt.legend(["Training Error", "Validation Error"])
plt.savefig("Error.png")

#Initialize B matrix for regularization
B = np.identity(10)
B[0][0] = 0

m=9
for lam in [-20, -4]: #Iterate over best and underfitting lambda
    w = np.dot(np.linalg.inv(A+(len(xTrain)/2*B*2*np.exp(lam))), c) #Compute W based on lambda

    regularizedTrainingOutput = computePrediction(xTrain, w, m) #Compute training regularized prediction
    plotModel(regularizedTrainingOutput, xTrain, m, w, tTrain, "Training Data, Regularization " + str(lam)) #Plot regularized training prediction

    regularizedValidationOutput = computePrediction(xValid, w, m) #Compute validation regularized prediction
    plotModel(regularizedValidationOutput, xValid, m, w, tValid, "Validation Data, Regularization " + str(lam)) #Plot regularized validation prediction

plt.show()