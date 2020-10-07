import numpy as np
import sklearn.datasets as skds
import matplotlib.pyplot as plt

x, target = skds.load_boston(return_X_y=True)
x = x.T
for index, feature in enumerate(x):
    plt.figure(index)
    plt.scatter(np.log(feature), target)
    plt.title("log(" + skds.load_boston().feature_names[index] + ") vs MEDV")
    plt.xlabel(skds.load_boston().feature_names[index])
    plt.ylabel("MEDV")
    plt.savefig(skds.load_boston().feature_names[index]+"log.png")
    