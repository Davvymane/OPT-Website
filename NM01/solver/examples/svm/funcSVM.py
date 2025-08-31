import numpy as np
from scipy.sparse import eye as speye

def funcSVM(x, key, w, A=None, c=None):
    n = len(x)
    if key == 'f':
        out = np.linalg.norm(x, 'fro')**2 - (1 - w) * x[n - 1]**2
    elif key == 'g':
        out = x.copy()
        out[n - 1] = w * x[n - 1]
    elif key == 'h':
        out = speye(n).toarray()
        out[n - 1, n - 1] = w
    elif key == 'a':
        acc = lambda var: np.count_nonzero(np.sign(A @ var[:n - 1] + var[n - 1]) - c)
        out = 1 - acc(x) / len(c)
    else:
        out = None  # 'Otherwise' is REQIURED
    return out