import numpy as np
import random
import math
import time
import heapq

from multiprocessing import Process, Queue

from Tools import *

from threading import Thread
import threading

random.seed(1998)

buffer_PC1 = Queue(10)  # Buffer for the first Producer-Consumer: R and R_test
buffer_PC2 = Queue(10)  # Buffer for the second Producer-Consumer: R_hat and R_test
# R_test is just forwarded, but simpler like that

NUMBER_OF_THREADS = 2
final_res = []


def producer(db, n_cross):
    global buffer_PC1

    split = load_indexes(db, n_cross)

    # Put values: (R, R_test)
    for v in range(n_cross):
        R = build_R_from_DB(split[:v] + split[v + 1:])
        R_test = build_R_from_DB([split[v]])
        buffer_PC1.put((R, R_test))

    # Final values to indicate the end of the
    for t in range(NUMBER_OF_THREADS):
        print("coucou")
        buffer_PC1.put(None)

    return


def compute(k):
    global buffer_PC1
    global buffer_PC2

    while True:
        value = buffer_PC1.get()
        if value is None:  # The end of the thread
            print("ok")
            buffer_PC2.put(None)  # Indicate its end
            return

        r, r_test = value  # Decompose the value into r and r_test
        r_hat = uBkNN(r, k)  # Compute r_hat from the function defined below
        buffer_PC2.put((r_hat, r_test))  # Put values: (r_hat, r_test)


def consumer(n_cross):
    global buffer_PC2

    counter_end = 0  # Counts the number of threads that are finished

    processed = 0

    MSE_g = np.zeros(n_cross)
    MAE_g = np.zeros(n_cross)

    while True:
        value = buffer_PC2.get()
        if value is None:
            counter_end += 1  # One more thread is finished
            if counter_end == NUMBER_OF_THREADS:  # All threads are done
                final_res.append((np.mean(MSE_g), np.mean(MAE_g)))  # Compute mean
                return
            continue  # Restart the loop

        r_hat, r_test = value  # Decomposes values
        print("get one element", processed)
        nrow, ncol = r_test.shape
        MSE = 0.0
        MAE = 0.0
        length = 0
        for i in range(nrow):
            for j in range(ncol):
                if r_test[i, j] != 0:
                    # print(R_test[i, j], R_hat[i, j])
                    MSE += (r_test[i, j] - r_hat[i, j]) ** 2
                    MAE += abs(r_test[i, j] - r_hat[i, j])
                    length += 1
        MSE /= length
        MAE /= length
        print(MSE, MAE)
        MSE_g[processed] = MSE
        MAE_g[processed] = MAE
        processed += 1


def uBkNN(r, k):

    # print('je rentre')
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
                a = a/np.sqrt(np.dot(r[i, :], r[j, :]))
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


if __name__ == "__main__":
    R, db = open_file("../ml-100k/u.data")
    k = 15
    n_cross = 10

    a = time.time()

    producer_thread = Process(target=producer, args=(db, n_cross))
    compute_threads = [Process(target=compute, args=(k,)) for i in range(NUMBER_OF_THREADS)]
    consumer_thread = Process(target=consumer, args=(n_cross,))

    producer_thread.start()
    for i in range(NUMBER_OF_THREADS):
        compute_threads[i].start()
    consumer_thread.start()

    producer_thread.join()
    for i in range(NUMBER_OF_THREADS):
        compute_threads[i].join()
    consumer_thread.join()

    print(final_res)
    print(time.time() - a)
