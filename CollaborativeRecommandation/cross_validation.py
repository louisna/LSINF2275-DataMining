from UBkNN import uBkNN
from UBkNN_sd import uBkNN_sd
from mf_sgd import sgd
from weighted_slope_one import weighted_slope_one
from basic_algorithms import normal_predictor, baseline

from surprise import Dataset
from surprise import SVD, SlopeOne, SVDpp, NormalPredictor, BaselineOnly, NMF, CoClustering
from surprise.model_selection import cross_validate

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time


def build_R_from_DB(DB, indexes):
    nb_users = 943
    nb_items = 1682

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    for i in indexes:
        user, item, rating, _ = DB[i, :]
        R[int(user) - 1, int(item) - 1] = int(rating)

    return R


def cross_validation(DB, k, n_folds=10, cf=uBkNN):
    MSE_g = np.zeros(n_folds)
    MAE_g = np.zeros(n_folds)

    # Using sklearn KFolds to create the 10 folds
    kf = KFold(n_splits=n_folds, random_state=1998, shuffle=True)

    index_fold = 0
    for train_index, test_index in kf.split(DB):
        R = build_R_from_DB(DB, train_index)
        R_test = build_R_from_DB(DB, test_index)
        # print(v)
        a = time.time()
        R_hat = cf(R)
        print(time.time() - a)
        nrow, ncol = R_test.shape
        MSE = 0.0
        MAE = 0.0
        for i in range(nrow):
            for j in range(ncol):
                if R_test[i, j] != 0:
                    # print(R_test[i, j], R_hat[i, j])
                    MSE += (R_test[i, j] - R_hat[i, j]) ** 2
                    MAE += abs(R_test[i, j] - R_hat[i, j])
        MSE /= len(test_index)
        MAE /= len(test_index)
        print(MSE, MAE, np.sqrt(MSE))
        MSE_g[index_fold] = MSE
        MAE_g[index_fold] = MAE
        index_fold += 1

    MSE = np.mean(MSE_g)
    MAE = np.mean(MAE_g)

    return MSE, MAE


def cross_validation_surprise():
    algos = [SVD(), SlopeOne(), SVDpp(), NormalPredictor(), BaselineOnly(), NMF(), CoClustering()]
    results = []

    DB = Dataset.load_builtin('ml-100k')

    for algo in algos:
        # Cross validation
        res = cross_validate(algo, data=DB, measures=['MSE', 'MAE'], cv=10)
        print(res)
        results.append(res)

    print(results)


# From http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
def open_file(filename):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(filename, sep='\t', names=r_cols, encoding='latin-1')

    return ratings.values


if __name__ == '__main__':
    DB = open_file('ml-100k/u.data')
    cross_validation(DB, 40, cf=baseline)
    # cross_validation_surprise()
