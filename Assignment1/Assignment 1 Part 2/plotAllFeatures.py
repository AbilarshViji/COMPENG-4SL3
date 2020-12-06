import numpy as np
import sklearn.datasets as skds
import matplotlib.pyplot as plt

x, target = skds.load_boston(return_X_y=True)
x = x.T
for index, feature in enumerate(x):
    plt.figure(index)
    fig, axs = plt.subplots(2, 2)
    axs[0,0].scatter(feature, target)
    axs[0,0].set_title("(" + skds.load_boston().feature_names[index] + ") vs MEDV")
    axs[0,1].scatter(np.log(feature), target)
    axs[0,1].set_title("log(" + skds.load_boston().feature_names[index] + ") vs MEDV")
    axs[1,0].scatter(np.power(feature, 2), target)
    axs[1,0].set_title("sq(" + skds.load_boston().feature_names[index] + ") vs MEDV")
    axs[1,1].scatter(np.sqrt(feature), target)
    axs[1,1].set_title("sqrt(" + skds.load_boston().feature_names[index] + ") vs MEDV")
    plt.xlabel(skds.load_boston().feature_names[index])
    plt.ylabel("MEDV")
    fig.tight_layout()
    plt.savefig(skds.load_boston().feature_names[index]+"ALL.png")
    