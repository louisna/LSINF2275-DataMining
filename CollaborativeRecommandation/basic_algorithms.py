import numpy as np
from tqdm import tqdm
import math


def normal_predictor(R):
    """
    Algorithm predicting a random rating based on the distribution of the training set, which
    is assumed tobe normal (surprise library doc)
    hat{r}_{u,i} generated from normal distribution (mu, sigma) where
    mu, sigma are generated from maximum likelihood estimation
    :param R: the rating matrix
    :return: prediction for missing values of R
    """

    n_users, n_movies = R.shape

    mu = np.sum(R) / np.count_nonzero(R)

    nz = np.nonzero(R)
    var = 0.0
    for ind in range(len(nz[0])):
        i = nz[0][ind]
        j = nz[1][ind]
        var += (R[i, j] - mu) ** 2

    sigma = np.sqrt(var / np.count_nonzero(R))

    R_hat = R.copy().astype(float)

    # Predict for non-rated movies
    for u in tqdm(range(n_users)):
        for i in range(n_movies):
            if R[u, i] != 0:
                continue
            R_hat[u, i] = np.random.normal(loc=mu, scale=sigma)

    return R_hat


def baseline(R):
    """
    Algorithm predicting the baseline estimate for given user and item (surprise library doc)
    :param R: the rating matrix
    :return: prediction for missing values of R
    """

    R_hat = R.copy().astype(float)

    n_users, n_movies = R.shape

    mu = np.sum(R) / np.count_nonzero(R)

    # Compute user bias
    # Compute mean for each user
    # The bias is the difference between the global mean and the user mean
    means = np.true_divide(R.sum(1), (R != 0).sum(1))
    means = np.array([0.0 if math.isnan(i) else i for i in means])
    bias_users = means - mu

    # Compute movie bias
    # Compute mean for each movie
    means_m = np.true_divide(R.T.sum(1), (R.T != 0).sum(1))
    means_m = np.array([0.0 if math.isnan(i) else i for i in means_m])
    # The bias is the difference between the global mean and the movie mean
    bias_movies = means_m - mu

    # Predict for non-rated movies
    for u in tqdm(range(n_users)):
        for i in range(n_movies):
            if R[u, i] != 0:
                continue
            R_hat[u, i] = mu + bias_users[u] + bias_movies[i]

    return R_hat
