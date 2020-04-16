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
    sim_matrix = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(i+1, n_row):
            a = np.dot(r[i, :], r[j, :])
            if a != 0.0:
                a = a/np.sqrt(np.dot(r[i, :], r[j, :]))
                sim_matrix[i, j] = a
                sim_matrix[j, i] = a

    threshold = np.sqrt(n_row)

    r_hat = r.copy()

    for i in range(n_row):
        print(i)
        a = [(sim_matrix[i, j], j) for j in range(n_row)]
        a.sort(key=lambda iii: -iii[0])
        for j in range(n_col):
            if r[i, j] != 0:  # Not compute if already exist
                continue
            if len(vertical[j][0]) == 0:  # In case no one purchased this item
                continue  # Useless to try to compute it

            if len(vertical[j][0]) <= k:  # Every elements of vertical will be used
                kNN = [(sim_matrix[i, zz], zz) for zz in vertical[j][0]]  # if sim_matrix[i, zz] > 0.0 ?
            elif len(vertical[j][0]) < threshold:  # If not much elements: can do like before
                kNN = []
                # print("process")
                for client in vertical[j][0]:
                    sim = sim_matrix[i, client]
                    if len(kNN) < k:
                        heapq.heappush(kNN, (sim, client))
                    elif len(kNN) >= k and kNN[0][0] < sim:  # Full kNN and update
                        heapq.heappop(kNN)
                        heapq.heappush(kNN, (sim, client))
            else:
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
            #else:
            #    print('-----------------')
            #    print(vertical[j][0], j, i)
            #    print(sim_matrix[i, vertical[j][0][0]])

    return r_hat


def split_ratings(DB, cross=10):
    split = []
    nrow, _ = DB.shape

    total = list(range(nrow))

    for i in range(cross - 1):
        print(i)
        length = round(len(total) / (cross - i))
        index_sample = random.sample(range(len(total)), length)
        split.append([total[j] for j in index_sample])
        total = [total[j] for j in range(len(total)) if j not in index_sample]
    split.append(list(total))
    split_DB = []
    for spl in split:
        a = [DB[i, :] for i in spl]
        split_DB.append(a)
    return split_DB


def load_indexes(DB, cross=10, filepath="ml-100k/u.data"):
    filepath = filepath[:-5] + "_indexes.data"
    try:
        with open(filepath, "r") as fd:
            split = []
            n = []
            for line in fd:
                if line == '\n':
                    if n:
                        split.append(n)
                    n = []
                    continue
                a = line.split('\t')
                n.append(np.array([float(a[0]), float(a[1]), float(a[2])]))
            if n:
                split.append(n)
            return split
    except IOError:
        print("No such file", filepath)
    split_DB = split_ratings(DB, cross)
    with open(filepath, "w+") as fd:
        for s in split_DB:
            for i in s:
                fd.write("" + str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\t" + str(1234567890) + "\n")
            fd.write('\n')
            print('ok')
    print("rip")
    return split_DB


def build_R_from_DB(splits):
    # nb_ratings = 100000
    nb_ratings = 20000
    nb_users = 943
    nb_items = 1682

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    for split in splits:
        for user, item, rating in split:
            R[int(user) - 1, int(item) - 1] = int(rating)

    return R


def cross_validation(DB, k, n_cross=10):
    MSE_g = np.zeros(10)
    MAE_g = np.zeros(10)

    split = load_indexes(DB, n_cross)

    for v in range(n_cross):
        # v is the testo set, splitted[-v] is the training set
        R = build_R_from_DB(split[:v] + split[v + 1:])
        R_test = build_R_from_DB([split[v]])
        # print(v)
        a = time.time()
        R_hat = uBkNN(R, k)
        print(time.time() - a)
        # print('\n\n')
        # R_hat = dummy(R)
        nrow, ncol = R_test.shape
        MSE = 0.0
        MAE = 0.0
        for i in range(nrow):
            for j in range(ncol):
                if R_test[i, j] != 0:
                    # print(R_test[i, j], R_hat[i, j])
                    MSE += (R_test[i, j] - R_hat[i, j]) ** 2
                    MAE += abs(R_test[i, j] - R_hat[i, j])
        MSE /= len(split[v])
        MAE /= len(split[v])
        print(MSE, MAE)
        MSE_g[v] = MSE
        MAE_g[v] = MAE

    MSE = np.mean(MSE_g)
    MAE = np.mean(MAE_g)

    return MSE, MAE


def open_file(filepath):
    """
    The pattern in the file is the following:
    user id | item id | rating | timestamp
    """
    nb_ratings = 100000
    # nb_ratings = 20000
    nb_users = 943
    nb_items = 1682

    DB = np.zeros((nb_ratings, 3))

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    with open(filepath, "r") as fd:
        count = 0
        for line in fd:
            if not line or line == "\n":
                continue
            user_id, item_id, rating, timestamp = list(line.split('\t'))
            R[int(user_id) - 1, int(item_id) - 1] = int(rating)
            DB[count] = np.array([int(user_id), int(item_id), int(rating)])
            count += 1

    return R, DB


if __name__ == "__main__":
    R, DB = open_file("ml-100k/u.data")
    cross_validation(DB, 15)
