import numpy as np

def refine(x, sp, A0=None, c=None):
    # x:  a vector in R^{n x 1}               (REQUIRED)
    # sp: a positive integer in [1,n] or [].  (REQUIRED)
    # AO: a matrix in R^{m x n}. If sp ~=[], then A0 is REQUIRED
    # c:  a vector in R^{m x 1}. If sp ~=[], then c  is REQUIRED

    x = x.flatten() # guarantee x is a 1-D array
    m, n = A0.shape
    if sp is not None and len(sp) > 0:
        K = 6
        Ts = np.argsort(np.abs(x))[::-1][:sp + K - 1]
        sx = np.sort(np.abs(x))[::-1][:sp + K - 1]
        HD = np.ones(K)
        X = np.zeros((n, K))
        if sx[sp - 1] - sx[sp] <= 5e-2:
            tem = Ts[sp - 1:]
            for i in range(K):
                X[:, i] = np.zeros(n)
                X[Ts[:sp - 1], i] = x[Ts[:sp - 1]]
                X[tem[i], i] = x[tem[i]]
                X[:, i] = X[:, i] / np.linalg.norm(X[:, i])
                HD[i] = np.count_nonzero(np.sign(A0 @ X[:, i]) - c) / m
            i = np.argmin(HD)
            refx = X[:, i]
        else:
            refx = np.zeros(n)
            refx[Ts[:sp]] = x[Ts[:sp]] / np.linalg.norm(x[Ts[:sp]])
    else:
        refx = SparseApprox(x)
        refx = refx / np.linalg.norm(refx)

    if np.isnan(refx).any():
        refx = SparseApprox(x)
        refx = refx / np.linalg.norm(refx)

    return refx

# get the sparse approximation of x----------------------------------------
def SparseApprox(x0):
    n = len(x0)
    x = np.abs(x0[np.abs(x0) > 1e-2 / n])
    sx = np.sort(x[x != 0])
    if len(sx) < 2:
        return x0

    ratio = sx[1:] / sx[:-1]
    norm_ratio = (ratio - np.min(ratio)) / (np.max(ratio) - np.min(ratio) + 1e-12)
    mx = np.max(norm_ratio)
    it = np.where(norm_ratio == mx)[0]

    th = 0
    if mx > 10 and it[0] > 0:
        th = sx[it[0]]
    x0[np.abs(x0) <= th] = 0
    return x0