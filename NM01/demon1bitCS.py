# Solving 1 bit compressive sensing using randomly generated data

from func1bcs import func1BCS
from refine import refine
from NM01 import NM01
from plotting import plot_recovery
from datageneration1bcs import datageneration1BCS

import numpy as np

n = 1000
m = int(np.ceil(0.5 * n))
s = int(np.ceil(0.01 * n))            # sparsity level
r = 0.01                              # flipping ratio
nf = 0.05                             # noisy ratio

A, c, co, xo = datageneration1BCS('Ind', m, n, s, nf, r, 0.5)  # data generation

func = lambda x, key: func1BCS(x, key, 1e-5, 0.5, A, c)
B = (-c[:, None]) * A
b = (n * 8e-5) * np.ones((m, 1))
lam = 1

pars = {}
pars['tau'] = 1
pars['strict'] = (n <= 2000)

out = NM01(func, B, b, lam, pars)
x = refine(out['sol'], s, A, c)

plot_recovery(xo, x, pos=(950, 500, 500, 250), show_info=True)

print(' Computational time:    %.3fsec' % out['time'])
print(' Signal-to-noise ratio: %.2f' % (-20 * np.log10(np.linalg.norm(x - xo))))
print(' Hamming distance:      %.3f' % (np.count_nonzero(np.sign(A @ x) - c) / m))
print(' Hamming error:         %.3f' % (np.count_nonzero(np.sign(A @ x) - co) / m))