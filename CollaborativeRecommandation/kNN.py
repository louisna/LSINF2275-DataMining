import numpy as np
import sys
import random
from queue import PriorityQueue
# first ligne == user  second colonne == item

result = np.array([[1, 0, 3],
                   [2, 0, 5],
                   [0, 2, 0],
                   [2, 1, 2],
                   [3, 3, 3],
                   [2, 0, 5],])

def kNN(R,k): # R = numpy matix
    # calculing user's vector V[i] = vi in the slides
    V = [] # list de user
    num_rows, num_cols = R.shape
    for i in range(num_rows):
        v = np.array([0.0]*num_cols)
        for j in range(num_cols):
            if R[i,j] > 0:
                v[j] = 1
            else :
                v[j] = 0
        V.append(v)
    #print(V)

    # computing the k nearest neighbours
    nearest_neighbours = [] # nearest_neighbours[i] == k nearest neighbours of user i
    for i in range(num_rows):
        kbestPQ = PriorityQueue() # (similitude, number of client)
        for ii in range(num_rows):
            if i != ii :
                sim = sim_cosine(V[i],V[ii])
                if kbestPQ.qsize() < k :
                    kbestPQ.put((sim, ii))
                    continue
                # len(kbest) == k
                e = kbestPQ.get() # O(1) instead O(nlogn)
                if sim > e[0] :
                    kbestPQ.put((sim, ii)) # O(logn) instead O(1)
                else :
                    kbestPQ.put(e) # O(logn) instead O(1)
        kbest = []
        while(not kbestPQ.empty()):
            kbest.append(kbestPQ.get())
        nearest_neighbours.append(kbest)

    # computing the prediction
    R_hat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            denom = 0
            pred = 0
            for e in nearest_neighbours[i]: # here PQ is inefficient because must creat a new PQ
                denom += e[0]
                pred += e[0] * R[e[1],j]
            pred /= denom
            R_hat[i,j] = pred

    return R_hat

def sim_cosine(i,p): # i and p are numpy verctors i ==
    return np.dot(i.T,p)/(len(i)*len(p))


def split_ratings(DB, cross=10):

    split = []
    nrow, _ = DB.shape

    total = list(range(nrow))

    for i in range(cross-1):
        length = round(len(total)/(cross-i))
        index_sample = random.sample(range(len(total)), length)
        split.append([total[j] for j in index_sample])
        total = [total[j] for j in range(len(total)) if j not in index_sample]
    split.append(list(total))
    split_DB = []
    for spl in split:
        a = [DB[i,:] for i in spl]
        split_DB.append(a)
    return split_DB


def build_R_from_DB(splits):
    nb_ratings = 100000
    # nb_ratings = 23
    nb_users = 943
    nb_items = 1682

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    for split in splits:
        for user, item, rating in split:
            R[int(user)-1, int(item)-1] = int(rating)
    
    return R


def cross_validation(DB, k, n_cross=10):
    MSE_g = np.zeros(10)
    MAE_g = np.zeros(10)

    split = split_ratings(DB, n_cross)

    for v in range(n_cross):
        # v is the testo set, splitted[-v] is the training set
        R = build_R_from_DB(split[:v] + split[v+1:])
        R_test = build_R_from_DB([split[v]])
        R_hat = kNN(R, k)
        nrow, ncol = R_test.shape
        MSE = 0.0
        MAE = 0.0
        for i in range(nrow):
            for j in range(ncol):
                MSE += (R_test[i,j] - R_hat[i,j]) ** 2
                MAE += abs(R_test[i,j] - R_hat[i,j])
        MSE /= len(split[v])
        MAE /= len(split[v])
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
    # nb_ratings = 23
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
            R[int(user_id)-1, int(item_id)-1] = int(rating)
            DB[count] = np.array([int(user_id), int(item_id), int(rating)])
            count += 1

    return R, DB


if __name__ == "__main__":
    R, DB = open_file("ml-100k/u.data")
    #R_hat = kNN(R,2)
    #num_rows, num_cols = R_hat.shape
    #print(R_hat)
    #print(R_hat[:,num_cols-1])
    #print(split_ratings(R))
    print(cross_validation(DB, 3))
    # print(split_ratings(100, cross=40))
