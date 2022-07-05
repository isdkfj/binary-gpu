import numpy as np
from numba import jit

# create a enumeration matrix
def create_enum(d):
    pw = 2 ** d
    a = np.array(range(pw))
    res = np.zeros((d, pw))
    for i in range(d):
        res[i, ((a >> i) & 1) == 1] = 1
    return res[:, 1:]

def leverage_score_sampling(A, k):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    l = np.sum(u ** 2, axis=1)
    p = l / np.sum(l)
    row = np.random.choice(A.shape[0], k, replace=True, p=p)
    p = p.reshape(A.shape[0], 1)
    S = A[row, :] / np.sqrt(k * p[row, :])
    enum = create_enum(k)
    r = enum / np.sqrt(k * p[row, :])
    sol = np.linalg.solve(np.dot(S.T, S), np.dot(S.T, r))
    return sol

@jit(nopython=True)
def global_minl2(A, x):
    A = A.astype(np.float32)
    x = x.astype(np.float32)
    x = np.ascontiguousarray(x.T)
    cov = np.dot(A.T, A)
    sol, val = None, None
    for i in range(x.shape[0]):
        b = np.dot(A, x[i, :])
        b = (b > 0.5)
        if b.mean() < 0.4:
            continue
        b = b.astype(np.float32)
        y = np.linalg.solve(cov, np.dot(A.T, b))
        b = np.dot(A, y)
        l2 = np.sum(np.minimum(b ** 2, (b - 1.) ** 2))
        if val is None or l2 < val:
            sol, val = y, l2
    return sol, val

def leverage_score_solve(A, it, k):
    sol, val = None, None
    # run several iterations
    for i in range(it):
        x = leverage_score_sampling(A, k)
        p, v = global_minl2(A, x)
        if val is None or (v is not None and v < val):
            sol, val = p, v
    return sol, val

@jit(nopython=True)
def check_binary(A, x):
    A = A.astype(np.float32)
    x = x.astype(np.float32)
    x = np.ascontiguousarray(x.T)
    ans = []
    for i in range(x.shape[0]):
        b = np.dot(A, x[i, :])
        l2 = np.max(np.minimum(b ** 2, (b - 1.) ** 2))
        if l2 < 1e-6:
            print(l2)
            ans.append((b > 0.5).astype(np.float32))
    return ans

def equality_solve(A):
    d = A.shape[1]
    row = np.random.choice(A.shape[0], d, replace=False)
    S = A[row, :]
    while np.linalg.matrix_rank(S) < d:
        row = np.random.choice(A.shape[0], d, replace=False)
        S = A[row, :]
    enum = create_enum(d)
    sol = np.linalg.solve(S, enum)
    return check_binary(A, sol)
