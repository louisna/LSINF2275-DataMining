import numpy as np
from tqdm import tqdm


def weighted_slope_one(R):
    n_users, n_movies = R.shape

    sets = np.array([set(np.nonzero(R[:, i])[0]) for i in range(n_movies)], ndmin=2, dtype=set)
    S = np.array(sets.T & sets)

    # Vertical may contain empty values
    seen_by_users = []
    for u in range(n_users):
        seen_by_users.append(np.nonzero(R[u, :]))

    dev = np.zeros(shape=S.shape)

    # TODO: improve this loop
    for i in range(n_movies):
        for j in range(n_movies):
            num = 0
            for u in S[i, j]:
                num += R[u, i] - R[u, j]
            if len(S[i, j]) == 0:
                dev[i, j] = 0.0
            else:
                dev[i, j] = num / len(S[i, j])

    R_hat = R.copy()

    # TODO: improve this loop
    for u in tqdm(range(n_users)):
        for i in range(n_movies):
            if R[u, i] != 0:
                continue  # Pass

            num = 0.0
            den = 0.0
            for j in seen_by_users[u][0]:
                num += (dev[i, j] + R[u, j]) * len(S[i, j])
                den += len(S[i, j])

            if den == 0:
                R_hat[u, i] = 0.0
            else:
                R_hat[u, i] = num / den

    return R_hat


# TODO: Per-user trustiness Approaches

# TODO: Item-usefulness based Approaches
def weighted_slope_one_item_usefulness(R):
    n_users, n_movies = R.shape

    sets = np.array([set(np.nonzero(R[:, i])[0]) for i in range(n_movies)], ndmin=2, dtype=set)
    S = np.array(sets.T & sets)

    # Vertical may contain empty values
    seen_by_users = []
    for u in range(n_users):
        seen_by_users.append(np.nonzero(R[u, :]))

    dev = np.zeros(shape=S.shape)

    # TODO: improve this loop
    for i in range(n_movies):
        for j in range(n_movies):
            num = 0
            for u in S[i, j]:
                num += R[u, i] - R[u, j]
            if len(S[i, j]) == 0:
                dev[i, j] = 0.0
            else:
                dev[i, j] = num / len(S[i, j])

    MAE = np.zeros(shape=R.shape)

    for u in tqdm(range(n_users)):
        for i in range(n_movies):
            num = 0.0
            for i_prime in seen_by_users[u][0]:
                num += abs(R[u, i] + dev[i_prime, i] - R[u, i_prime])
            den = len(seen_by_users[u][0])
            if den == 0:
                MAE[u, i] = 0
            else:
                MAE[u, i] = num/den

    item_usefulness = np.zeros(shape=R.shape)
    max_MAE = np.amax(MAE)
    # item_usefulness = max_MAE - MAE  # Linear
    for u in tqdm(range(n_users)):
        for i in range(n_movies):
            item_usefulness[u, i] = 30 ** (max_MAE - MAE[u, i])  # Exponential


    R_hat = R.copy()
    # TODO: improve this loop
    for u in tqdm(range(n_users)):
        for i in range(n_movies):
            if R[u, i] != 0:
                continue  # Pass

            num = 0.0
            den = 0.0
            for j in seen_by_users[u][0]:
                num += (dev[i, j] + R[u, j]) * len(S[i, j]) * item_usefulness[u, j]
                den += len(S[i, j]) * item_usefulness[u, j]

            if den == 0:
                R_hat[u, i] = 0.0
            else:
                R_hat[u, i] = num / den

    return R_hat
