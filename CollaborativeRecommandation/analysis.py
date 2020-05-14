import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import random as rnd
import time
from cross_validation import *
from UBkNN import uBkNN
from UBkNN_sd import uBkNN_sd
from tqdm import tqdm
import pandas as pd


# To show standard deviation caps
matplotlib.rcParams.update({'errorbar.capsize': 3})

def from_algo_to_string(cf):
    if cf == uBkNN:
        return 'ubknn'
    elif cf == uBkNN_sd:
        return 'ubknn_sd'
    else:
        return 'itemrank'


def analyze_by_k(DB, cf, min_k=1, max_k=30):
    """
    Analyzes the accuracy of the the algorithm cf
    """
    k = list(range(min_k, max_k+1))

    filename = "analyze_k_{}_{}_{}.txt".format(min_k, max_k, from_algo_to_string(cf))

    with open(filename, "a+") as fd:
        for i in tqdm(k):
            MSE, MAE = cross_validation(DB, i, cf=cf, analyzing=True)
            out = "" + str(i) + " "
            for j in MSE:
                out += str(j) + " "
            for j in MAE:
                out += str(j) + " "
            out += "\n"
            fd.write(out)
            fd.flush()


def open_file_k(filename):
    res = pd.read_csv(filename, sep=' ', encoding='latin-1')
    res_np = res.values
    k = res_np[:, 0]  # First column of the dataframe contains k
    MSE = res_np[:, 1:11]  # Next 10 columns are values of the MSE
    MAE = res_np[:, 11:]  # Last 10 columns are values of the MAE
    return k, MSE, MAE


def plot_analyze_k(filename):
    k, MSE, MAE = open_file_k(filename)

    MSE_mean = np.mean(MSE, axis=1)
    MAE_mean = np.mean(MAE, axis=1)

    MSE_sd = np.std(MSE, axis=1)
    MAE_sd = np.std(MAE, axis=1)

    print(MSE_sd)

    plt.errorbar(k, MSE_mean, yerr=MSE_sd, fmt='-o', label='UBkNN with sd scaling')
    plt.ylabel('MSE')
    plt.xlabel('Number of neighbors')
    plt.title('User-based kNN on ml-100k')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    # DB = open_file('ml-100k/u.data')
    # analyze_by_k(DB, uBkNN_sd, 1, 50)
    # analyze_by_k(DB, uBkNN, 1, 50)
    plot_analyze_k('analyze_k_1_30_ubknn_sd.txt')