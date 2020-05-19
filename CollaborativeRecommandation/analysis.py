import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import random as rnd
import time
from cross_validation import *
from UBkNN import uBkNN
from UBkNN_sd import uBkNN_sd
from mf_sgd import sgd
from weighted_slope_one import weighted_slope_one, weighted_slope_one_item_usefulness
from basic_algorithms import normal_predictor, baseline, baseline_biased
from tqdm import tqdm
import pandas as pd


# To show standard deviation caps
matplotlib.rcParams.update({'errorbar.capsize': 3})

def from_algo_to_string(cf):
    if cf == uBkNN:
        return 'ubknn'
    elif cf == uBkNN_sd:
        return 'ubknn_sd'


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

    # Take only 1/2
    k = k[::2]
    ksd = ksd[::2]
    MSE = MSE[::2]
    MAE = MAE[::2]
    MSEsd = MSEsd[::2]
    MAEsd = MAEsd[::2]

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

    plt.figure()
    plt.errorbar(k, MSE_mean, yerr=MSE_sd, fmt='-o', label='UBkNN', alpha=0.7)
    plt.errorbar(ksd, MSE_meansd, yerr=MSE_sdsd, fmt='-x', label='UBkNN with sd scaling')
    plt.ylabel('MSE')
    plt.xlabel('Number of neighbors')
    plt.grid(True)
    plt.title('Evolution of the MSE of the UBkNN with k')
    plt.legend(loc='best')
    plt.savefig('ubknn_k_MSE.svg')
    plt.show()

    plt.figure()
    plt.errorbar(k, MAE_mean, yerr=MAE_sd, fmt='-o', label='UBkNN', alpha=0.7)
    plt.errorbar(ksd, MAE_meansd, yerr=MAE_sdsd, fmt='-x', label='UBkNN with sd scaling')
    plt.ylabel('MAE')
    plt.xlabel('Number of neighbors')
    plt.grid(True)
    plt.title('Evolution of the MAE of the UBkNN with k')
    plt.legend(loc='best')
    plt.savefig('ubknn_k_MAE.svg')

    plt.show()


def retrieve_surprise_results():
    results_MSE, results_MAE = cross_validation_surprise()

    with open('results_surprise_MSE.txt', 'w+') as mse_file:
        for k in results_MSE.keys():
            values = results_MSE[k]
            to_write = " ".join(map(str, values))
            mse_file.write(k + " " + to_write + '\n')

    with open('results_surprise_MAE.txt', 'w+') as mae_file:
        for k in results_MAE.keys():
            values = results_MAE[k]
            to_write = " ".join(map(str, values))
            mae_file.write(k + " " + to_write + '\n')


def values_from_surprise():
    # MSE
    df = pd.read_csv('results_surprise_MSE.txt', header=None, delimiter=" ")
    df_np = df.values
    names = df_np[:, 0]
    values = df_np[:, 1:]

    print(values)

    means = np.mean(values, axis=1)
    a = np.asarray(values).astype(np.float64)
    std = np.std(a, axis=1)

    print(names)
    print(means)
    print(std)

    # MAE
    df = pd.read_csv('results_surprise_MAE.txt', header=None, delimiter=" ")
    df_np = df.values
    names = df_np[:, 0]
    values = df_np[:, 1:]

    print(values)

    means = np.mean(values, axis=1)
    a = np.asarray(values).astype(np.float64)
    std = np.std(a, axis=1)

    print(names)
    print(means)
    print(std)


def analyze_models(DB):
    algos = [uBkNN, uBkNN_sd, sgd, weighted_slope_one, weighted_slope_one_item_usefulness, baseline, baseline_biased, normal_predictor]
    res_MSE = np.zeros(shape=(len(algos), 10))
    res_MAE = np.zeros(shape=(len(algos), 10))

    for i in tqdm(range(len(algos))):
        algo = algos[i]
        res_MSE[i], res_MAE[i] = cross_validation(DB, cf=algo, analyzing=True)
    # np.savetxt('comparison_algos_MSE.txt', X=res_MSE, delimiter=',')
    # np.savetxt('comparison_algos_MAE.txt', X=res_MAE, delimiter=',')


# https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
def analyse_models_output():
    file_MSE = 'comparison_algos_MSE.txt'
    file_MAE = 'comparison_algos_MAE.txt'

    val_MSE = np.loadtxt(file_MSE, delimiter=',')
    val_MAE = np.loadtxt(file_MAE, delimiter=',')

    n_row, n_col = val_MSE.shape

    mean_MSE = np.zeros((n_row))
    mean_MAE = np.zeros((n_row))
    sd_MSE = np.zeros((n_row))
    sd_MAE = np.zeros((n_row))
    min_MSE = np.zeros(n_row)
    min_MAE = np.zeros(n_row)
    max_MSE = np.zeros(n_row)
    max_MAE = np.zeros(n_row)

    for i in range(n_row):
        mean_MSE[i] = np.mean(val_MSE[i, :])
        sd_MSE[i] = np.std(val_MSE[i, :])
        min_MSE[i] = np.min(val_MSE[i, :])
        max_MSE[i] = np.max(val_MSE[i, :])

        mean_MAE[i] = np.mean(val_MAE[i, :])
        sd_MAE[i] = np.std(val_MAE[i, :])
        min_MAE[i] = np.min(val_MAE[i, :])
        max_MAE[i] = np.max(val_MAE[i, :])

    print("MSE m", mean_MSE)
    print("MSE sd", sd_MSE)
    print("MAE m", mean_MAE)
    print("MAE sd", sd_MAE)

    # MSE
    fig, ax = plt.subplots()
    ax.errorbar(np.arange(6), mean_MSE[:6], sd_MSE[:6], fmt='ok', lw=3)
    ax.errorbar(np.arange(6), mean_MSE[:6], [mean_MSE[:6] - min_MSE[:6], max_MSE[:6] - mean_MSE[:6]],
                 fmt='.k', ecolor='gray', lw=1)
    ax.set_xticks(np.arange(6))
    xticks = ['UBkNN', "UBkNN (sd)", "SGD", "WSO", "WSO (IU)", "Baseline"]
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_ylabel('MSE')
    ax.set_title('MSE of the different algorithms')
    ax.yaxis.grid(True)
    plt.savefig("results_algos_MSE.svg")
    plt.show()

    # MAE
    fig, ax = plt.subplots()
    ax.errorbar(np.arange(6), mean_MAE[:6], sd_MAE[:6], fmt='ok', lw=3)
    ax.errorbar(np.arange(6), mean_MAE[:6], [mean_MAE[:6] - min_MAE[:6], max_MAE[:6] - mean_MAE[:6]],
                fmt='.k', ecolor='gray', lw=1)
    ax.set_xticks(np.arange(6))
    xticks = ['UBkNN', "UBkNN (sd)", "SGD", "WSO", "WSO (IU)", "Baseline"]
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_ylabel('MAE')
    ax.set_title('MAE of the different algorithms')
    ax.yaxis.grid(True)
    plt.savefig("results_algos_MAE.svg")
    plt.show()


if __name__ == '__main__':
    DB = open_file('ml-100k/u.data')
    # analyze_by_k(DB, uBkNN_sd, 1, 50)
    # analyze_by_k(DB, uBkNN, 1, 50)
    # plot_analyze_k('analyze_k_1_50_ubknn.txt', 'analyze_k_1_50_ubknn_sd.txt')
    # retrieve_surprise_results()
    analyze_models(DB)
    # analyse_models_output()
    # values_from_surprise()
    # a = cross_validation(DB, cf=baseline, analyzing=True)
    # print(a)
