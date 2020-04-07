import numpy as np
import sys
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

def min(kbest):
    min_sim = sys.maxsize
    index = -1
    for i in range(len(kbest)):
        elem = kbest[i]
        if elem[0] < min_sim:
            min_sim = elem[0]
            index = i
    return (min_sim, index)




def open_file(filepath):
    """
    The pattern in the file is the following:
    user id | item id | rating | timestamp
    """
    nb_ratings = 100000
    nb_users = 943
    nb_items = 1682

    # Create the rating matrix
    R = np.zeros((nb_users, nb_items))

    with open(filepath, "r") as fd:
        for line in fd:
            if not line or line == "\n":
                continue
            user_id, item_id, rating, timestamp = list(line.split('\t'))
            R[int(user_id)-1, int(item_id)-1] = int(rating)

    return R


if __name__ == "__main__":
    R = open_file("ml-100k/u.data")
    R_hat = kNN(result,2)
    num_rows, num_cols = R_hat.shape
    print(R_hat)
    print(R_hat[:,num_cols-1])