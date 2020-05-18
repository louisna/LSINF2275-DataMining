from UBkNN import uBkNN
from UBkNN_sd import uBkNN_sd
from mf_sgd import sgd
from weighted_slope_one import weighted_slope_one, weighted_slope_one_item_usefulness
from basic_algorithms import normal_predictor, baseline

from surprise import Dataset
from surprise import SVD, SlopeOne, SVDpp, NormalPredictor, BaselineOnly, NMF, CoClustering
from surprise.model_selection import cross_validate

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time
from tqdm import tqdm


def build_R_from_DB(DB, indexes):
    """
    Build the rating matrix from the given database, and the retrieved indexes
    :param DB: the complete database
    :param indexes: the indexes of the ratings that will be put in the rating matrix
    :return: the rating matrix
    """
    nb_users = 943  # Static for the ml-100k
    nb_items = 1682  # Static for the ml-100k

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    for i in indexes:
        user, item, rating, _ = DB[i, :]
        R[int(user) - 1, int(item) - 1] = int(rating)

    return R


def cross_validation(DB, k, n_folds=10, cf=uBkNN, analyzing=False):
    """
    Performs a n_folds-folds cross validation on the given collaborative filtering algorithm (cf)
    :param DB: the complete database
    :param k: a parameter, used in some models (default:40)
    :param n_folds: the number of folds (default: 10)
    :param cf: the collaborative filtering algorithm (default: uBkNN)
    :return: the averaged MSE and MAE on the 10 folds
    """
    MSE_g = np.zeros(n_folds)
    MAE_g = np.zeros(n_folds)

    # Using sklearn KFolds to create the 10 folds
    kf = KFold(n_splits=n_folds, random_state=1998, shuffle=True)

    index_fold = 0
    for train_index, test_index in kf.split(DB):
        R = build_R_from_DB(DB, train_index)  # Train
        R_valid = build_R_from_DB(DB, test_index)  # Validation
        a = time.time()
        R_hat = cf(R)
        # print(time.time() - a)
        nrow, ncol = R_valid.shape
        MSE = 0.0
        MAE = 0.0
        for i in range(nrow):
            for j in range(ncol):
                if R_valid[i, j] != 0:
                    # print(R_test[i, j], R_hat[i, j])
                    MSE += (R_valid[i, j] - R_hat[i, j]) ** 2
                    MAE += abs(R_valid[i, j] - R_hat[i, j])
        MSE /= len(test_index)
        MAE /= len(test_index)
        # print(MSE, MAE)
        print(MSE, MAE, np.sqrt(MSE))  # Also print the RMSE for comparison with other algorithms
        MSE_g[index_fold] = MSE
        MAE_g[index_fold] = MAE
        index_fold += 1

    MSE = np.mean(MSE_g)
    MAE = np.mean(MAE_g)

    if analyzing:
        return MSE_g, MAE_g

    return MSE, MAE


def cross_validation_surprise():
    """
    Performs a 10-folds cross-validation on algorithms from the surprise library. It is used to compare the performance
    of our models with those implemented by the library
    :return: ???
    """
    algos = [SVD(), SlopeOne(), SVDpp(), NormalPredictor(), BaselineOnly(), NMF(), CoClustering()]
    algos_name = ['SVD', 'SlopeOne', 'SVDpp', 'NormalPredictor', 'BaselineOnly', 'NMF', 'CoClustering']
    results_MSE = {}
    results_MAE = {}

    DB = Dataset.load_builtin('ml-100k')

    for i in tqdm(range(len(algos))):
        # Cross validation
        res = cross_validate(algos[i], data=DB, measures=['MSE', 'MAE'], cv=10)
        print(res)
        results_MSE[algos_name[i]] = res['test_mse']
        results_MAE[algos_name[i]] = res['test_mae']

    return results_MSE, results_MAE


# From http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
def open_file(filename):
    """
    Open the file filename and returns an numpy.array containing the ratings
    :param filename: the path of the file
    :return: numpy.array containing the ratings
    """
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(filename, sep='\t', names=r_cols, encoding='latin-1')

    return ratings.values


if __name__ == '__main__':
    DB = open_file('ml-100k/u.data')
    # cross_validation(DB, 40, cf=weighted_slope_one_item_usefulness)
    cross_validation_surprise()
