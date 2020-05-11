import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
import random

random.seed(1998)


def als(r, k=30):
    ### https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    alpha = 40
    lam = 1e-3
    nb_iteration = 50

    # data(R) must be csr_matrix format n by m
    r_hat = r.copy()
    r = csr_matrix(r)
    n_row, n_col = r.shape

    # init
    X = csr_matrix(np.random.normal(size=(n_row, k)))
    Y = csr_matrix(np.random.normal(size=(n_col, k)))

    X_I = eye(n_row)
    Y_I = eye(n_col)

    I = eye(k)
    lI = lam * I

    # confidence level
    c = csr_matrix(r.copy() * alpha)

    # iterate
    for iteration in range(nb_iteration):

        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        print(iteration)
        for u in range(n_row):  # for each user u
            u_row = c[u, :].toarray()
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            CuI = diags(u_row, [0])
            Cu = CuI + Y_I

            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

        for i in range(n_col):  # for each item i
            i_row = c[:, i].T.toarray()

            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            CiI = diags(i_row, [0])
            Ci = CiI + X_I

            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    r_approx = X.dot(Y.T)

    for i in range(n_row):
        for j in range(n_col):
            if r_hat[i, j] != 0:  # Not compute if already exist
                continue
            r_hat[i, j] = r_approx[i, j]

    return r_hat
