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
    res = res.replace(r'\\n', ' ', regex=True)
    res_np = res.values
    k = res_np[:, 0]  # First column of the dataframe contains k
    MSE = res_np[:, 1:11]  # Next 10 columns are values of the MSE
    MAE = res_np[:, 11:21]  # Last 10 columns are values of the MAE
    return k, MSE, MAE


def plot_analyze_k(filename, filename_sd):
    k, MSE, MAE = open_file_k(filename)
    ksd, MSEsd, MAEsd = open_file_k(filename_sd)

    # Values of the UBkNN
    MSE_mean = np.mean(MSE, axis=1)
    MAE_mean = np.mean(MAE, axis=1)

    MSE_sd = np.std(MSE, axis=1)
    MAE_sd = np.std(MAE, axis=1)

    # Now the values of the UBkNN sd
    MSE_meansd = np.mean(MSEsd, axis=1)
    MAE_meansd = np.mean(MAEsd, axis=1)

    MSE_sdsd = np.std(MSEsd, axis=1)
    MAE_sdsd = np.std(MAEsd, axis=1)

    plt.subplot(121)
    plt.errorbar(k, MSE_mean, yerr=MSE_sd, fmt='-o', label='UBkNN')
    plt.errorbar(ksd, MSE_meansd, yerr=MSE_sdsd, fmt='-x', label='UBkNN with sd scaling')
    plt.ylabel('MSE')
    plt.xlabel('Number of neighbors')
    plt.grid(True)
    plt.title('Evolution of the MSE of the UBkNN with k')
    plt.legend(loc='best')
    plt.savefig('ubknn_k.svg')

    plt.subplot(122)
    plt.errorbar(k, MAE_mean, yerr=MAE_sd, fmt='-o', label='UBkNN')
    plt.errorbar(ksd, MAE_meansd, yerr=MAE_sdsd, fmt='-x', label='UBkNN with sd scaling')
    plt.ylabel('MSE')
    plt.xlabel('Number of neighbors')
    plt.grid(True)
    plt.title('Evolution of the MSE of the UBkNN with k')
    plt.legend(loc='best')
    plt.savefig('ubknn_k.svg')

    plt.show()


if __name__ == '__main__':
    # DB = open_file('ml-100k/u.data')
    # analyze_by_k(DB, uBkNN_sd, 1, 50)
    # analyze_by_k(DB, uBkNN, 1, 50)
    plot_analyze_k('analyze_k_1_50_ubknn.txt', 'analyze_k_1_50_ubknn_sd.txt')