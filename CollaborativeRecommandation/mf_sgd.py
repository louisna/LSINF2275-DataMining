import numpy as np
from tqdm import tqdm
from weighted_slope_one import weighted_slope_one, weighted_slope_one_item_usefulness


# This algorithm is strongly inspired by the article available on
# https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea
def sgd(R, k=40, learning_rate=0.001, n_iter=80, lmbda=0.01, pre_compute_WSO=False):
    """
    Computes an explicit matrix factorization using stochastic gradient descent
    :param R: the rating matrix
    :param k: the number of latent features. Default=40, as advised in the article
    :param learning_rate: the learning rate, used during the SGD. Default=0.001, as advised in the article
    :param n_iter: number of iterations of SGD. Default=80, good trade-off between performances and cost
    :param lmbda: parameter of the L2 regularization. Default=0.01, arbitrarily set
    :param pre_compute_WSO: if True, pre-compute the missing values using the Weighted SlopeOne algorithm
                            it could improve the performances of the final result, but is much more costly
    :return: the rating prediction matrix
    """

    if pre_compute_WSO:
        R = weighted_slope_one_item_usefulness(R)

    def predict(u, i):
        """
        Predicts the score of user u on movie i with the current X and Y vectors
        :param u: user index
        :param i: movie index
        :return: the predicted value
        """
        pred = global_bias + user_bias[u] + movie_bias[i]
        pred += X[u, :].dot(Y[i, :].T)
        return pred

    n_users, n_movies = R.shape

    # Init: random values following a normal distribution, with standard deviation = 1/k
    X = np.random.normal(size=(n_users, k), scale=1./k)
    Y = np.random.normal(size=(n_movies, k), scale=1./k)

    user_bias = np.zeros(n_users)
    movie_bias = np.zeros(n_movies)
    global_bias = np.mean(R[np.where(R != 0)])

    u_non_zero, i_non_zero = np.nonzero(R)

    # Repeat the process for n_iter iterations
    for iteration in tqdm(range(n_iter)):

        # Repeat for each sample
        for index in range(len(u_non_zero)):
            u = u_non_zero[index]
            i = i_non_zero[index]
            prediction = predict(u, i)

            # Compute the error
            e = R[u, i] - prediction

            # Update
            user_bias[u] += learning_rate * (e - lmbda * user_bias[u])
            movie_bias[i] += learning_rate * (e - lmbda * movie_bias[u])

            # Update latent factors
            X[u, :] += learning_rate * (e * Y[i, :] - lmbda * X[u, :])
            Y[u, :] += learning_rate * (e * X[u, :] - lmbda * Y[i, :])

    # Once done, predict all non-rated
    R_hat = np.zeros(shape=(n_users, n_movies), dtype=float)
    for u in range(n_users):
        for i in range(n_movies):
            if R[u, i] != 0.0:
                continue
            R_hat[u, i] = predict(u, i)

    return R_hat
