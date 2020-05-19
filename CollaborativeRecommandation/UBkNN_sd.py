import numpy as np
import math
import heapq
from tqdm import tqdm


def uBkNN_sd(r, k=26):
    """
        Compute the rating predictions for missing values of r, using a user-based kNN model
        It also takes into account the standard deviation of the users to predicted more accurately
        :param r: the rating matrix
        :param k: the number of neighbors. Default = 26, found empirically
        :return: the rating prediction matrix
        """
    n_row, n_col = r.shape
    # Compute vertical representation of R

    # Vertical may contain empty values
    vertical = []
    for j in range(n_col):
        vertical.append(np.nonzero(r[:, j]))

    # Compute mean for each user
    means = np.true_divide(r.sum(1), (r != 0).sum(1))

    means = [0.0 if math.isnan(i) else i for i in means]

    # Compute standard deviation
    standard_deviation = [np.nanstd(np.where(np.isclose(a, 0), np.nan, a)) for a in r]

    # Compute sim matrix
    sim_matrix = np.zeros((n_row, n_row))
    for i in range(n_row):
        for j in range(i+1, n_row):
            a = np.dot(r[i, :], r[j, :])
            if a != 0.0:
                a = a/(np.linalg.norm(r[i, :]) * np.linalg.norm(r[j, :]))
                sim_matrix[i, j] = a
                sim_matrix[j, i] = a

    # Threshold to speed-up the computation time
    # if, for a movie j, there is less than 'threshold' users that rated this movie,
    # perform a classic search among all users that have rated this movie
    # otherwise: there are a lot of people that rated this movie
    # it is then faster, for user i, to check his neighbors in decreasing order of similarity score
    # and retrieve the user if it has rated this movie (we have a high chance of that, since the number of people
    # that rated this movie is high)
    threshold = k * 3

    r_hat = r.copy().astype(float)

    for i in tqdm(range(n_row)):
        # Sort the users according to similarity score
        # Used later if high number of users that rated a movie
        a = [(sim_matrix[i, j], j) for j in range(n_row)]
        a.sort(key=lambda iii: iii[0], reverse=True)
        for j in range(n_col):
            if r[i, j] != 0:  # Not compute if already exist
                continue
            if len(vertical[j][0]) == 0:  # In case no one purchased this item
                continue  # Useless to try to compute it

            if len(vertical[j][0]) <= k:  # Every elements of vertical[j] will be used
                kNN = [(sim_matrix[i, zz], zz) for zz in vertical[j][0]]
            elif len(vertical[j][0]) < threshold:  # If not much rating users: search in that set
                kNN = []
                # Simple heap search
                for client in vertical[j][0]:
                    sim = sim_matrix[i, client]
                    if len(kNN) < k:
                        heapq.heappush(kNN, (sim, client))
                    elif len(kNN) >= k and kNN[0][0] < sim:  # Full kNN and update
                        heapq.heapreplace(kNN, (sim, client))
            else:  # Search first in the most similar users, those who rated this item
                kNN = []
                for (sim, other) in a:
                    if r[other, j] > 0.0:
                        kNN.append((sim, other))
                    if len(kNN) == k:
                        break

            # We have here the kNN of user i (if at least k)
            pred = 0.0
            den = 0.0
            for sim, client in kNN:
                pred += sim * (r[client, j] - means[client]) / standard_deviation[client]
                den += abs(sim)
            if den != 0:  # 0 similarity: could happen
                pred /= den

                r_hat[i, j] = standard_deviation[i] * pred + means[i]

    return r_hat
