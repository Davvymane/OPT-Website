# Solving support vector machine using real datasets

import numpy as np
import scipy.io as sio
from funcSVM import funcSVM
from NM01 import NM01
from normalization import normalization

test = 1
data = ['arce', 'fabc']
prob = data[test]

samp = sio.loadmat(f'{prob}.mat')         
label = sio.loadmat(f'{prob}_class.mat')  

A = normalization(samp['X'], 2)
c = label['y'].flatten()
c[c != 1] = -1
m, n0 = A.shape

func = lambda x, key: funcSVM(x, key, 1e-4, A, c)
B = (-c[:, None]) * np.hstack((A, np.ones((m, 1))))
b = np.ones((m, 1))

pars = {}
pars['tau'] = 1
lam = 10

out = NM01(func, B, b, lam, pars)
acc = 1 - np.count_nonzero(np.sign(np.hstack((A, np.ones((m, 1)))) @ out['sol']) - c) / m

print(' Training  Size:       %d x %d' % (m, n0))
print(' Training  Time:       %5.3fsec' % out['time'])
print(' Training  Accuracy:   %5.2f%%' % (acc * 100))
print(' Training  Objective:  %5.3e' % out['obj'])