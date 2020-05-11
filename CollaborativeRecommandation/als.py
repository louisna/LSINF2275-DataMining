import numpy as np
from tqdm import tqdm
import math
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# This algorithm is based on the article from
# https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
# https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

# TODO: Once the convergence criterion is done: increase max_iter to 50 (or more ?)
def cf_als(R, k, lbd=0.06, max_iter=10, alpha=1):
    """
    Collaborative filtering algorithm using matrix factorization
    :param R: The training matrix of ratings
    :param k: number of latent dimensions
    :param lbd: lambda parameter for the regularization
    :param max_iter: maximal number of iterations
    :param alpha: scaling parameter for the preference importance
    :return: R_hat, the matrix containing the predictions
    """

    n_users, n_movies = R.shape

    # Init X and Y to random values:
    # One vector x for each user, one vector y for each movie
    # Each vector of dimension k (k latent classes)
    X = np.random.normal(size=(n_users, k), loc=2.5)
    Y = np.random.normal(size=(n_movies, k), loc=2.5)

    # Confidence matrix
    C = R.copy()
    C = alpha * C

    # Pre-compute I and lbd * I
    X_I = np.eye(n_users)
    Y_I = np.eye(n_movies)
    Id = np.eye(k)
    lI = lbd * Id

    # TODO: include a convergence criterion to stop the iterations
    # Repeat until convergence / maximum number of iterations reached
    for iteration in tqdm(range(max_iter)):

        # Pre-compute X.T * X and Y.T * Y
        xtx = np.dot(X.T, X)
        yty = np.dot(Y.T, Y)

        # First update the x's, keep Y constant
        for u in range(n_users):
            # Get the user row
            u_row = C[u, :]

            # Preference of user u
            p_u = u_row.copy()

            # Compute Cu and Cu - I
            # TODO: check if good behavior
            CuI = np.diag(u_row)
            Cu = CuI + Y_I

            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = np.linalg.solve(yty + yT_CuI_y + lI, yT_Cu_pu)
            X[u, X[u] < 0] = 0.0

        # Second update the y's, keep X constant
        for i in range(n_movies):
            # Get the movie column
            i_row = C[:, i].T

            # Preference for that movie
            p_i = i_row.copy()

            # Compute Ci and Ci - I
            CiI = np.diag(i_row)
            Ci = CiI + X_I

            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = np.linalg.solve(xtx + xT_CiI_x + lI, xT_Ci_pi)
            Y[i, Y[i] < 0] = 0.0

    return np.dot(X, Y.T)


def test(R, k):
    model = ALS.train(R, k, 10)
    pred = model.predictAll(R)
    return pred


if __name__ == '__main__':
    a = np.array([
        [1, 2, 4, 2],
        [1, 2, 4, 2],
        [1, 2, 0, 2]])
    print(test(a, 2))
