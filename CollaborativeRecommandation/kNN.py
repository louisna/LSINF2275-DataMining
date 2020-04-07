import numpy as np
import sys

# first ligne == user  second colonne == item

result = np.array([[11, 12, 13],
                   [21, 22, 23],
                   [31, 32, 33]])

def kNN(R,k): # R = numpy matix
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

    nearest_neighbours = []
    for i in range(num_rows):
        kbest = [] # (similitude, number of client)
        for ii in range(num_rows):
            if i != ii :
                kbest.sort( key=lambda t: t[0])
                sim = sim_cosine(V[i],V[ii])
                if len(kbest) >= k and sim > kbest[0][0] :
                    kbest[0] = (sim, ii)
                elif len(kbest) < k :
                    kbest.append((sim, ii))
        nearest_neighbours.append(kbest)

    R_hat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            denom = 0
            pred = 0
            for e in nearest_neighbours[i]:
                denom += e[0]
                pred += e[0] * R[e[1],j]
            pred /= denom
            R_hat[i,j] = pred

    return R_hat

def sim_cosine(i,p): # i and p are numpy verctors i ==
    return np.dot(i.T,p)/(len(i)*len(p))








def open_file(filepath):
    """
    The pattern in the file is the following:
    user id | item id | rating | timestamp
    """
    nb_ratings = 100000
    nb_users = 943
    nb_items = 1982

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
    R_hat = kNN(R,10)
    num_rows, num_cols = R_hat.shape
    print(R_hat)
    print(R_hat[:,num_cols-1])
