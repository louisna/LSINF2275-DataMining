import numpy as np
import sys
import random
import math
from queue import PriorityQueue
import time
import heapq
random.seed(1998)


def uBkNN(r, k):
    n_row, n_col = r.shape
    # Compute vertical representation of R

    # Vertical may contain empty values
    vertical = []
    for j in range(n_col):
        vertical.append(np.nonzero(r[:, j]))

    # Compute mean for each user
    means = np.true_divide(r.sum(1), (r != 0).sum(1))

    means = [0.0 if math.isnan(i) else i for i in means]

    # Compute sim matrix
    sim_matrix = np.zeros((n_row, n_row))
    for i in range(n_row):
        for j in range(i+1, n_row):
            a = np.dot(r[i, :], r[j, :])
            if a != 0.0:
                a = a/(np.linalg.norm(r[i, :]) * np.linalg.norm(r[j, :]))
                sim_matrix[i, j] = a
                sim_matrix[j, i] = a

    threshold = k * 3

    r_hat = r.copy()

    for i in range(n_row):
        a = [(sim_matrix[i, j], j) for j in range(n_row)]
        a.sort(key=lambda iii: -iii[0])
        for j in range(n_col):
            if r[i, j] != 0:  # Not compute if already exist
                continue
            if len(vertical[j][0]) == 0:  # In case no one purchased this item
                continue  # Useless to try to compute it

            if len(vertical[j][0]) <= k:  # Every elements of vertical will be used
                kNN = [(sim_matrix[i, zz], zz) for zz in vertical[j][0]]  # if sim_matrix[i, zz] > 0.0 ?
            elif len(vertical[j][0]) < threshold:  # If not much rating users: search in that set
                kNN = []
                for client in vertical[j][0]:
                    sim = sim_matrix[i, client]
                    if len(kNN) < k:
                        heapq.heappush(kNN, (sim, client))
                    elif len(kNN) >= k and kNN[0][0] < sim:  # Full kNN and update
                        heapq.heappop(kNN)
                        heapq.heappush(kNN, (sim, client))
            else:  # Search first in the most promising users, those who rated this item
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
                pred += sim * (r[client, j] - means[client])
                den += abs(sim)
            if den != 0:  # 0 similarity: could happen
                pred /= den

                r_hat[i, j] = pred + means[i]

    return r_hat
