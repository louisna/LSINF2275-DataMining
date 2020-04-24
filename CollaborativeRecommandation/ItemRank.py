import numpy as np
from Tools import *
import time

epsilon = 10 ** -6


def compute_correlation_matrix_user_preferences(R):
    n_row, n_col = R.shape
    sets = np.array([set(np.nonzero(R[:, i])[0]) for i in range(n_col)], ndmin=2, dtype=set)
    U = sets.T & sets
    len_vec = np.vectorize(len)
    for i in range(n_col):  # Maybe improve this loop
        U[i, i] = set()
    C_tilde = len_vec(U)
    omega = np.array([np.sum(C_tilde[:, i]) for i in range(n_col)])
    omega2 = np.array([1. if i == 0 else i for i in omega])
    C = C_tilde / omega2

    # TODO: Improve computation of D
    den = np.array([np.sum(R[i, :]) for i in range(n_row)])
    den = np.array([1. if i == 0 else i for i in den])
    D = np.array([R[i, :] / den[i] for i in range(n_row)])
    return C, D.T


def itemRank(R, alpha=0.85, doa=False):
    n_row, n_col = R.shape
    print("compute C and D")
    C, D = compute_correlation_matrix_user_preferences(R)
    print("C and D done")
    print(C)
    R_hat = R.copy()

    IR_global = np.zeros((n_row, n_col))

    for i in range(n_row):
        IR_before = np.ones(n_col) / n_col  # Initial IR

        IR = alpha * np.dot(C, IR_before) + (1 - alpha) * D[:, i]
        # Repeat until convergence
        count = 1
        while np.linalg.norm(IR - IR_before) > epsilon:
            IR_before = IR
            IR = alpha * np.dot(C, IR_before) + (1 - alpha) * D[:, i]
            count += 1
        print(i, count)
        # Replace value in R_hat
        for j in range(n_col):
            if R_hat[i, j] == 0:
                R_hat[i, j] = IR[j]
        if doa:
            IR_global[i, :] = IR
    if doa:
        return IR_global
    return R_hat


def cross_validation(DB, n_cross=10):
    MSE_g = np.zeros(10)
    MAE_g = np.zeros(10)

    split = load_indexes(DB, n_cross)

    for v in range(n_cross):
        # v is the testo set, splitted[-v] is the training set
        R = build_R_from_DB(split[:v] + split[v + 1:])
        R_test = build_R_from_DB([split[v]])
        # print(v)
        a = time.time()
        R_hat = itemRank(R)
        print(time.time() - a)
        # print('\n\n')
        # R_hat = dummy(R)
        nrow, ncol = R_test.shape
        MSE = 0.0
        MAE = 0.0
        for i in range(nrow):
            print("change user")
            print(np.sum(R_hat[i, :]))
            for j in range(ncol):
                if R_test[i, j] != 0:
                    # print(R_test[i, j], R_hat[i, j])
                    print(R_test[i, j], R_hat[i, j])
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


def compute_NW(R):
    n_row, n_col = R.shape
    NW = np.array([np.where(R[i, :] == 0) for i in range(n_row)])
    return NW


def compute_DAO(R, R_train, R_test):
    n_row, n_col = R_train.shape
    IR_global = itemRank(R_train, doa=True)
    print(IR_global[0, :])
    print('---------------')
    print(IR_global[1, :])
    print('---------------')
    print(IR_global[2, :])
    print('---------------')
    print(IR_global[3, :])
    print('---------------')
    print(IR_global[4, :])
    print('---------------')
    print(IR_global[5, :])
    NW = compute_NW(R)

    def check_order(i, j, k):
        if IR_global[i, j] >= IR_global[i, k]:
            return 1.
        return 0.

    DOA = np.array([0.0] * n_row)
    for i in range(n_row):
        Tui = np.nonzero(R_test[i, :])[0]
        NWui = NW[i, :][0]
        DOAui = 0.0
        for j in Tui:
            for k in NWui:
                DOAui += check_order(i, j, k)
        den = (len(Tui) * len(NWui))
        if den != 0.0:
            DOAui = DOAui / den
        else:
            DOAui = 0.0
        DOA[i] = DOAui

    macro_DOA = np.sum(DOA) / n_row
    print(macro_DOA)


if __name__ == "__main__":

    a = np.array([
        [1., 0., 2., 5., 0.],
        [0., 0., 1., 0., 5.],
        [1., 2., 3., 4., 5.],
        [0., 5., 0., 0., 0.]
    ])
    """
    print(itemRank(a))

    R, DB = open_file("ml-100k/u.data")
    cross_validation(DB)
    """
    R = R_from_filename("ml-100k/u.data")
    R_train = R_from_filename("ml-100k/u1.base")
    R_test = R_from_filename("ml-100k/u1.test")
    compute_DAO(R, R_train, R_test)
    # print(compute_NW(a))
    # compute_DAO(a, a)



