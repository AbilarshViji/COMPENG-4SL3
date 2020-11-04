import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

dataset = pd.read_csv('spambase.data')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values
rand = 1727
xTrain, xTest, tTrain, tTest = train_test_split(X, t, test_size = 1/3, random_state = rand)

def decisionTree(xTrain, xTest, tTrain, tTest, rand):
    errors = []
    best = 0
    bestError = 1
    for leaf in range(2, 401):
        dtc = DecisionTreeClassifier(random_state=rand, max_leaf_nodes=leaf)
        dtc.fit(xTrain, tTrain)
        error = mean_squared_error(tTest, dtc.predict(xTest))
        if bestError > error:
            best = leaf
            bestError = error
        errors.append(error)
    return errors, best

def baggingClassifiers(xTrain, xTest, tTrain, tTest, rand):
    errors = []
    for pred in range(50, 2501, 50):
        bc = BaggingClassifier(random_state=rand, base_estimator=DecisionTreeClassifier(), n_estimators=pred, n_jobs=6, verbose=1)
        bc.fit(xTrain, tTrain)
        error = mean_squared_error(tTest, bc.predict(xTest))
        errors.append(error)
    return errors

def randomForest(xTrain, xTest, tTrain, tTest, rand):
    errors = []
    for pred in range(50, 2501, 50):
        rfc = RandomForestClassifier(random_state=rand, n_estimators=pred, n_jobs=8, verbose=1) #base_estimator=DecisionTreeClassifier() is an attribute of RandomForestClassifier
        rfc.fit(xTrain, tTrain)
        error = mean_squared_error(tTest, rfc.predict(xTest))
        errors.append(error)        
    return errors

def adaBoostDecisionStump(xTrain, xTest, tTrain, tTest, rand):
    errors = []
    for pred in range(50, 2501, 50):
        abc = AdaBoostClassifier(random_state=rand, base_estimator=DecisionTreeClassifier(random_state=rand, max_depth=1), n_estimators=pred)
        abc.fit(xTrain, tTrain)
        error = mean_squared_error(tTest, abc.predict(xTest))
        errors.append(error)    
    return errors

def adaBoostDecisionTree10Leaf(xTrain, xTest, tTrain, tTest, rand):
    errors = []
    for pred in range(50, 2501, 50):
        abc = AdaBoostClassifier(random_state=rand, base_estimator=DecisionTreeClassifier(random_state=rand, max_leaf_nodes=10), n_estimators=pred)
        abc.fit(xTrain, tTrain)
        error = mean_squared_error(tTest, abc.predict(xTest))
        errors.append(error)
    return errors

def adaBoostDecisionTree(xTrain, xTest, tTrain, tTest, rand):
    errors = []
    for pred in range(50, 2501, 50):
        abc = AdaBoostClassifier(random_state=rand, base_estimator=DecisionTreeClassifier(random_state=rand), n_estimators=pred)
        abc.fit(xTrain, tTrain)
        error = mean_squared_error(tTest, abc.predict(xTest))
        errors.append(error)
    return errors

decisionTreeError, bestNumLeaf = decisionTree(xTrain, xTest, tTrain, tTest, rand)
plt.figure(2)
plt.plot(range(2, 401), decisionTreeError, color='g')
plt.savefig("DecisionTree.png")
baggingClassifierError = baggingClassifiers(xTrain, xTest, tTrain, tTest, rand)
plt.figure(3)
plt.plot(range(50, 2501, 50), baggingClassifierError, color='g')
plt.savefig("BaggingError.png")
randomForestError = randomForest(xTrain, xTest, tTrain, tTest, rand)
plt.figure(4)
plt.plot(range(50, 2501, 50), randomForestError, color='g')
plt.savefig("RandomForest.png")
adaBoostStumpError = adaBoostDecisionStump(xTrain, xTest, tTrain, tTest, rand)
plt.figure(5)
plt.plot(range(50, 2501, 50), adaBoostStumpError, color='g')
plt.savefig("AdaboostStump.png")
adaBoost10LeafError = adaBoostDecisionTree10Leaf(xTrain, xTest, tTrain, tTest, rand)
plt.figure(6)
plt.plot(adaBoost10LeafError, color='g')
plt.savefig("Adaboost10Leaf.png")
adaBoostError = adaBoostDecisionTree(xTrain, xTest, tTrain, tTest, rand)
plt.figure(7)
plt.plot(range(50, 2501, 50), adaBoostError, color='g')
plt.savefig("AdaboostError.png")

plt.figure(0, figsize = (9, 6))
plt.hlines(decisionTreeError[bestNumLeaf-2], 0, 2500, color='r')
plt.plot(range(50, 2501, 50), baggingClassifierError, color='g')
plt.plot(range(50, 2501, 50), randomForestError, color='b')
plt.plot(range(50, 2501, 50), adaBoostStumpError, color='c')
plt.plot(range(50, 2501, 50), adaBoost10LeafError, color='m')
plt.plot(range(50, 2501, 50), adaBoostError, color='y')

plt.xlabel("Number of Predictors")
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error vs Number of Predictors")
plt.legend(["Bagging Classifier", "Random Forest Classifier", "Adaboost with decision stump", "Adaboost with decision trees and max 10 leafs", "Adaboost with decision trees", "Decision Tree"], bbox_to_anchor=(1, 0.5), loc='upper left')
plt.tight_layout()
plt.savefig("testErrors.png")

plt.figure(1)
plt.plot(range(2, 401), decisionTreeError)
plt.xlabel("Number of Leafs")
plt.ylabel("Mean Squared Error")
plt.title("Cross-Validation Error vs Number of Leafs")
plt.savefig("CVE.png")
plt.show()