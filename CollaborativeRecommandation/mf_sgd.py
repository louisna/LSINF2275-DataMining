import numpy as np
from tqdm import tqdm


# https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea
def sgd(R, k, learning_rate=0.001, n_iter=100, lmbda=0.01):

    def predict(u, i):
        pred = global_bias + user_bias[u] + movie_bias[i]
        pred += X[u, :].dot(Y[i, :].T)
        return pred

    n_users, n_movies = R.shape

    X = np.random.normal(size=(n_users, k), scale=1./k)
    Y = np.random.normal(size=(n_movies, k), scale=1./k)

    user_bias = np.zeros(n_users)
    movie_bias = np.zeros(n_movies)
    global_bias = np.mean(R[np.where(R != 0)])

    # TODO: check if correct
    u_non_zero, i_non_zero = np.nonzero(R)

    for iteration in tqdm(range(n_iter)):

        # TODO: check if correct
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

    # Once done, predict all
    R_hat = np.zeros(shape=(n_users, n_movies))
    for u in range(n_users):
        for i in range(n_movies):
            R_hat[u, i] = predict(u, i)

    return R_hat
