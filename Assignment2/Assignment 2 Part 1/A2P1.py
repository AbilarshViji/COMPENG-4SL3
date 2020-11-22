import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, precision_recall_curve, plot_precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import random

def logisticReg(xTrain, xTest, tTrain, tTest, rand): #scikit-learn logistic regression implementation
    lr = LogisticRegression(random_state=rand, fit_intercept=True) #initialize model
    lr.fit(xTrain, tTrain) #fit model
    plot_precision_recall_curve(lr, xTest, tTest) #plot PR curve
    plt.title("Logistic Regression PR Curve")
    return lr.coef_, f1_score(tTest, lr.predict(xTest)), 1-accuracy_score(tTest, lr.predict(xTest)) #return f1 score and misclassification rate

def logisticRegImplement(xTrain, xTest, tTrain, tTest, alpha): #returns weights of logistic regression
    xTrainD = addDummy(xTrain)
    numSamples = xTrain.shape[0]
    iterations = 100 #number of iterations (tunable)
    w = [0]*xTrainD.shape[1] #starting weights (tunable)
    grNorms = np.zeros(iterations)
    for step in range(iterations):
        z = np.dot(xTrainD, w)
        y = 1/(1+np.exp(-z))
        diff = y - tTrain
        gr = np.dot(xTrainD.T, diff)/numSamples
        grNormSq = np.dot(gr, gr)
        grNorms[step] = grNormSq
        w -= alpha * gr #weight calculation with alpha (tunable)
    return w

def precisionRecallPred(w, xTest, tTest): #plots PR curve, and returns predictions mapped to sigmoid
    xTestD = addDummy(xTest) #add dummy
    pred = np.dot(xTestD, w) #compute prediction
    precision, recall, _  = precision_recall_curve(tTest, pred) #obtain precision and recall
    plt.plot(recall, precision) #plot precision and recall
    pred = np.round(1/(1+np.exp(-pred))) #map points to sigmoid, and round to make binary
    return pred

def f1ScoreAndMisclassification(actual, pred): #computes f1 score and misclassification rate
    tp, fp, fn =  0,0,0
    for a, p in zip(actual, pred): #iterate through actual and predicted value
        if a == 1 and p == 1: #true positive
            tp += 1
        elif p == 1 and a == 0: #false positive
            fp += 1
        elif p == 0 and a == 1: #false negative
            fn += 1
    p = tp / (tp+fp) #precision
    r = tp / (tp+fn) #recall
    f1 = (2*p*r) / (p+r) #f1 score
    return f1, (fp+fn)/len(pred) #return f1 score and misclassification rate

def kNN(xTrain, xTest, tTrain, tTest, rand, k): #scikit-learn knn implementation
    errors = []
    kf = KFold(n_splits=k, shuffle=True, random_state=1727).split(xTrain) #initialize kFolds
    for K, (train, test) in zip(range(1, k+1), kf):
        knn = KNeighborsClassifier(n_neighbors=K) #init classifier
        knn.fit(xTrain[train], tTrain[train]) #train classifier
        errors.append(1-accuracy_score(tTrain[test], knn.predict(xTrain[test]))) #add misclassification rate to list
    print("scikit-learn KkN misclassification rate", errors) #print all misclassification errors
    best = errors.index(min(errors)) + 1 #get best k with lowest misclassification rate
    knn = KNeighborsClassifier(n_neighbors=best) #init classifier with best K
    knn.fit(xTrain, tTrain) #train classifier
    return best, f1_score(tTest, knn.predict(xTest)), 1-accuracy_score(tTest, knn.predict(xTest)) #return best K, f1 score, and misclassification rate 

def KFoldImplement(folds, data): #my kfold implememtation with shuffling
    testShuffle = list(range(len(data))) #create list with indexes
    random.shuffle(testShuffle) #shuffle list
    testSize = len(data)//folds #get size of test set
    full = set(range(len(data))) #initialize set with indexes (sets are subtractable)
    ret = []
    while(testShuffle): #while there are still values that can become test sets
        test = testShuffle[:testSize] #create test set based on size defined above
        testShuffle = testShuffle[testSize:] #remove those values from testShuffle
        train = list(full-set(test)) #the values not in test will in train
        ret.append([train, test]) #add both these values to a fold
    return ret

def addDummy(array): #add dummy to data
    array = array.T
    ret = np.vstack((np.ones(array.shape[1]), array))
    return ret.T

def distance(train, test): #calculate distance between points
    return np.sum(np.absolute(train-test))

def getNeighbours(xTrain, xTestPt, k): #get neighbours for 1 point
    distances = []
    for i, trainPt in enumerate(xTrain): #iterate over all training points
        dist = distance(trainPt, xTestPt) #compute distance between training and testing point
        distances.append((i, dist)) #add index and distance to list
    distances.sort(key=lambda x:x[1]) #sort list by distances
    return [distances[x][0] for x in range(k)] #return k indexes with the shortest distances

def getPrediction(neighbours, tTrain): #get prediction for 1 point
    predictions = tTrain[neighbours] #get k predictions
    numOnes = np.count_nonzero(predictions) #count how many ones
    if numOnes > len(predictions) - numOnes: #if ones greates, return 1
        return 1
    else: #false negative safer then false positive
        return 0 #better to predict false in case of tie

def kNNImplement(xTrain, xTest, tTrain, tTest, k): #gets all predictions for certain k
    preds = []
    for x in xTest: #iterate over test points
        neighbours = getNeighbours(xTrain, x, k) #get neighbours for specific test point
        pred = getPrediction(neighbours, tTrain) #get prediction for test point
        preds.append(pred) #save prediction
    return f1ScoreAndMisclassification(tTest, preds) #use tess points, and predictions to compute f1 score and misclassification rate

def kNNComparison(xTrain, xTest, tTrain, tTest, k): #computes best K for values [1..k], and f1 score and misclassification rate for best K
    errors = []
    kf = KFoldImplement(5, xTrain) #initialize kfold
    for K, (train, test) in zip(range(1, k+1), kf):
        f1, error = kNNImplement(xTrain[train], xTrain[test], tTrain[train], tTrain[test], K) #get crossvalidation error for specific k
        errors.append(error) #save error
    print("My kNN misclassification rate", errors) #print all errors
    best = errors.index(min(errors)) + 1 #save best error
    return best, kNNImplement(xTrain, xTest, tTrain, tTest, best) #return best K, and f1 score and misclassification rate for best k

X, t = load_breast_cancer(return_X_y=True) #load data
rand = 1727
random.seed(rand) #random seed for my kfold
xTrain, xTest, tTrain, tTest = train_test_split(X, t, test_size = 1/3, random_state = rand) #split data

sc = StandardScaler()
xTrain[:, :]  = sc.fit_transform(xTrain[:, :])
xTest[:, :]  = sc.transform(xTest[:, :])

#scikit-learn logistic regression
w, sklLogf1, sklLogMcr = logisticReg(xTrain, xTest, tTrain, tTest, rand)
print("scikit-learn w values: ", w)
print("scikit-learn log F1 score: ", sklLogf1)
print("scikit-learn log Misclassification Rate: ", sklLogMcr)

#My logistic regression implementation
w = logisticRegImplement(xTrain, xTest, tTrain, tTest, 1)
print("My w values: ", w)
pred = precisionRecallPred(w, xTest, tTest)
myLogf1, myLogMcr = f1ScoreAndMisclassification(tTest, pred)
print("My log F1 score: ", myLogf1)
print("My log Misclassification Rate: ", myLogMcr)

#scikit-learn knn
sklBestK, sklkNNf1, sklkNNMcr = kNN(xTrain, xTest, tTrain, tTest, rand, 5)
print("scikit-learn Best K: ", sklBestK)
print("scikit-learn kNN F1 score: ", sklkNNf1)
print("scikit-learn kNN Misclassification Rate: ", sklkNNMcr)

#My knn implementation
myBestK, (mykNNf1, mykNNMcr) = kNNComparison(xTrain, xTest, tTrain, tTest, 5)
print("My Best K: ", myBestK)
print("My kNN F1 score: ", mykNNf1)
print("My kNN Misclassification Rate: ", mykNNMcr)

plt.legend(["scikit-learn PR", "My PR"])
plt.savefig("PRCurve.png")
plt.show()