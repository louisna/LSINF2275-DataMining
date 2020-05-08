import matplotlib.pyplot as plt

from UBkNN import *
from random import randrange, randint
import heapq

def similarity_score(R):

    n_row, n_col = R.shape

    # Compute sim matrix
    sim_matrix = np.zeros((n_row, n_row))
    for i in range(n_row):
        for j in range(i + 1, n_row):
            a = np.dot(R[i, :], R[j, :])
            if a != 0.0:
                b = a
                a = a/(np.linalg.norm(R[i, :]) * np.linalg.norm(R[j, :]))
                sim_matrix[i, j] = a
                sim_matrix[j, i] = a

    print("done 1")

    a = randrange(n_row)
    cmp1 = R[a, :]

    kNN = []

    for i in range(n_row):
        if i == a:
            continue
        score = sim_matrix[a, i]
        if len(kNN) < 1:
            heapq.heappush(kNN, (score, i))
        elif score > kNN[0][0]:
            heapq.heapreplace(kNN, (score, i))

    print(kNN)

    res1 = []
    res2 = []
    j = kNN[0][1]
    for i in range(n_row):
        if R[a, i] != 0 and R[j, i] != 0:
            res1.append(R[a, i])
            res2.append(R[j, i])

    return res1, res2

def aaaa(res1, res2):

    x = list(range(len(res1)))
    plt.scatter(x, res1)
    plt.scatter(x, res2)
    plt.show()

filename = 'ml-100k/u.data'
R, D = open_file(filename)
a, b = similarity_score(R)
aaaa(a, b)