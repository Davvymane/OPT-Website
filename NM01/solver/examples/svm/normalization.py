import numpy as np
import time
from scipy.sparse import diags

def normalization(X, normal_type):
    # This file aims to normalize the input matrix X;
    # Inputs:
    #       X           -- an (m x n) order matrix that will be normalized;
    #       normal_type -- type of normalization
    #                      =0; no normalization, i.e., NX=X;
    #                      =1; sample(row)-wise and then feature(column)-wise
    #                          normalization. This is better for the case when n<1000;
    #                      =2; feature(column)-wisely scaled to [-1,1], typically for logistic regression problem
    #                      =3; feature(column)-wisely scaled to unit norm columns, typically for CS problem
    # Outputs:
    #       NX           --  normalized m x n order matrix

    t0 = time.time()
    if normal_type == 0:  # No normalization
        NX = X

    elif normal_type == 1:
        C = X - np.mean(X, axis=1, keepdims=True)
        Yrow = C / np.std(X, axis=1, ddof=0, keepdims=True)  # Sample-wise  normalization
        Y = Yrow.T
        D = Y - np.mean(Y, axis=1, keepdims=True)
        Ycol = D / np.std(Y, axis=1, ddof=0, keepdims=True)  # Feature-wise normalization
        NX = Ycol.T
        if np.isnan(NX).any():
            nX = 1.0 / np.sqrt(np.sum(X * X, axis=0))
            lX = len(nX)
            NX = X @ diags(nX, offsets=0, shape=(lX, lX)).toarray()

    else:
        if normal_type == 2:
            nX = 1.0 / np.max(np.abs(X), axis=0)  # Feature-wisely scaled to [-1,1],
        else:
            nX = 1.0 / np.sqrt(np.sum(X * X, axis=0))  # Feature-wisely scaled to has unit norm columns,

        lX = len(nX)
        if lX <= 10000:
            NX = X @ diags(nX, offsets=0, shape=(lX, lX)).toarray()
        else:  # If lX is too large, seperate X into
            k = int(5e3)  # sveral smaller sub-matrices.
            if np.count_nonzero(X) / lX / lX < 1e-4:
                k = int(1e5)
            K = int(np.ceil(lX / k))
            for i in range(K - 1):
                T = np.arange(i * k, (i + 1) * k)
                X[:, T] = X[:, T] @ diags(nX[T], offsets=0, shape=(k, k)).toarray()
            T = np.arange((K - 1) * k, lX)
            k0 = len(T)
            X[:, T] = X[:, T] @ diags(nX[T], offsets=0, shape=(k0, k0)).toarray()
            NX = X

    NX[np.isnan(NX)] = 0
    print(" Nomorlization used %2.4f seconds." % (time.time() - t0))

    return NX