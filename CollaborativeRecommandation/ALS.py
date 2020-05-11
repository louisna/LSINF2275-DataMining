import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
import random

random.seed(1998)


def als(r, k=20):
    ### https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    lam = 1e-3
    nb_iteration = 100

    # data(R) must be csr_matrix format n by m
    r_hat = r.copy()
    r = csr_matrix(r)
    n_row, n_col = r.shape

    # init
    X = csr_matrix(np.random.normal(size=(n_row, k)))
    Y = csr_matrix(np.random.normal(size=(n_col, k)))

    # iterate
    for iteration in range(nb_iteration):

        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        print(iteration)
        for u in range(n_row):  # for each user u
            u_row = r[u, :].toarray()

            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            CuI = diags(u_row, [0])

            X[u] = spsolve(yTy + Y.T.dot(CuI).dot(Y) + lam * eye(k), Y.T.dot(CuI + eye(n_col)).dot(p_u.T))

        for i in range(n_col):  # for each item i
            i_row = r[:, i].T.toarray()

            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            CiI = diags(i_row, [0])

            Y[i] = spsolve(xTx + X.T.dot(CiI).dot(X) + lam * eye(k), X.T.dot(CiI + eye(n_row)).dot(p_i.T))

    r_approx = X.dot(Y.T)

    for i in range(n_row):
        for j in range(n_col):
            if r_hat[i, j] != 0:  # Not compute if already exist
                continue
            r_hat[i, j] = r_approx[i, j]

    return r_hat
