import numpy as np
import matplotlib.pyplot as plt

def computeModel(x, w, M):
    model = []
    for points in x:
        model.append(np.dot(w, points))
    return model

figCount = 0
def plotModel(model, x, M, w, train, title):
    global figCount
    plt.figure(figCount)
    figCount += 1
    plt.scatter(x, train, color="green")
    plt.title(title+ " Data, M = " + str(M))
    x = np.linspace(0.,1.,100)
    poly = []
    for val in x:
        temp = 0
        for m in range(M+1):
            temp += w[m]*(val**m)
        poly.append(temp)

    plt.plot(x, poly, color="cyan", linewidth=2)
    plt.plot(x, np.sin(4*np.pi*x), color="magenta", linewidth=2)
    plt.legend(["Predicted Function", "Actual Function", "Actual Points"])
    # plt.show()

def calcError(predictedList, actualList):
    error = 0
    for predict, actual in zip(predictedList, actualList):
        error += (predict-actual)**2

    return error/len(predictedList)

# N = 10, num training samples
# D = 0-9, num features
Xtrain = np.linspace(0.,1.,10) # training set
Xvalid = np.linspace(0.,1.,100) # validation set
np.random.seed(1727)
tvalid = np.sin(4*np.pi*Xvalid) + 0.3 * np.random.randn(100)
ttrain = np.sin(4*np.pi*Xtrain) + 0.3 * np.random.randn(10)

def computeTrainingData(maxM, Xtrain): #refactor XtrainD to smth better
    XtrainD = np.ones(len(Xtrain))
    if maxM > 0:
        XtrainD = np.vstack((XtrainD, Xtrain))
    for i in range(2, maxM + 1):
        temp = []
        for j in Xtrain:
            temp.append(j**i)
        XtrainD = np.vstack((XtrainD, temp))
    XtrainD = np.transpose(XtrainD)
    return XtrainD
maxM = 9

trainingError = []
validationError = []
m=0
XtrainD = computeTrainingData(m, Xtrain)
A = np.dot(np.transpose(XtrainD), XtrainD)
c = np.dot(np.transpose(XtrainD), ttrain)

w = [np.dot(1/A, c)]
model = computeModel(XtrainD, w, m)

plotModel(model, Xtrain, m, w, ttrain, "Training")
trainingError.append(calcError(model, ttrain))

XvalidD = computeTrainingData(m, Xvalid)
model = computeModel(XvalidD, w, m)
plotModel(model, Xvalid, m, w, tvalid, "Validation")
validationError.append(calcError(model, tvalid))
for m in range(1, maxM+1):
    XtrainD = computeTrainingData(m, Xtrain)
    A = np.dot(np.transpose(XtrainD), XtrainD)
    c = np.dot(np.transpose(XtrainD), ttrain)
    w = np.dot(np.linalg.inv(A), c)
   
    model = computeModel(XtrainD, w, m)

    plotModel(model, Xtrain, m, w, ttrain, "Training")
    trainingError.append(calcError(model, ttrain))

    XvalidD = computeTrainingData(m, Xvalid)
    model = computeModel(XvalidD, w, m)
    plotModel(model, Xvalid, m, w, tvalid, "Validation")
    validationError.append(calcError(model, tvalid))
plt.figure(figCount)
figCount += 1
plt.plot(trainingError,marker='o', linewidth=2)
plt.plot(validationError, marker='o', linewidth=2)
plt.title("Training and Validation Error")
plt.legend(["Training Error", "Validation Error"])
B = np.identity(10)
B[0][0] = 0
train = []
valid = []

m=9
for lam in [-33, -4]:
    w = np.dot(np.linalg.inv(A+(len(Xtrain)/2*B*2*np.exp(lam))), c)
    # print(w)
    regularizedModel = computeModel(XtrainD, w, m)
    train.append((lam, calcError(regularizedModel, ttrain)))
    regularizedValidationModel = computeModel(XvalidD, w, m)
    valid.append((lam, calcError(regularizedValidationModel, tvalid)))
    plt.figure(figCount)
    plotModel(regularizedModel, Xvalid, m, w, tvalid, "Regularization" + str(lam))
# print(sorted(train, key=lambda x:x[1]))
print(sorted(valid, key=lambda x:x[1]))


plt.show()

