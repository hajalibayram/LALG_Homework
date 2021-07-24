import numpy as np
from scipy import sparse
import copy
s=289963
np.random.seed(s)

def adj_mat(path):
    first = []
    second = []
    with open(path, 'r') as f:
        for line in f:
            l = line.strip()
            s = l.split()
            first.append(int(s[0]))
            second.append(int(s[1]))
        n = first[-1]+1 
    G = np.zeros((n, n))
    for i in range(len(first)):
        G[first[i]][second[i]] = 1
    return G, sparse.csr_matrix(G), n

def in_out_degs(G):
    r = np.sum(G, axis=1)
    c = np.sum(G, axis=0)
    return r, c

def hyperlink_mat(n ,G, c):
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if c[j] != 0:
                M[i][j] = G[i][j]/c[j]
    spM = sparse.csr_matrix(M)
    return M, spM

def mod_hyperlink_mat(n, G, c):
    M_hat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if c[j] == 0:
                M_hat[i][j] = float(1/n)
            else:
                M_hat[i][j] = G[i][j]/c[j]
    spM_hat = sparse.csr_matrix(M_hat)
    return M_hat, spM_hat


def power_method(A, tol=1e-16, maxIter=5000, x=None):
    n = A.shape[0]
    if x == None:
        x = np.ones((n,1))
    lambda_old = -1
    error = 1.0
    for i in range(maxIter):
        
        x = np.matmul(A,x)

        # Finding new Eigen value and Eigen vector
        lambda_new = max(abs(x))
        x = abs(x/lambda_new)

        error = abs(lambda_new-lambda_old)
        if error <= tol:
            break
        lambda_old = copy.deepcopy(lambda_new)

    return lambda_new, x/np.sum(x), i+1, error

def google_mat(G, c, alpha):
    n = G.shape[0]
    delta = (1-alpha)/n
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if c[j] == 0:
                A[i][j] = 1/n
            else:
                A[i][j] = alpha*(G[i][j]/c[j])+delta
    return A

def explore_web(A, maxIter=5000, tol=1e-16):
    n = A.shape[0]
    t = np.full((n,1), 1/n)
    A_tmp = copy.deepcopy(A)
    t_prev = copy.deepcopy(t)
    for k in range(maxIter):
        t = np.matmul(A_tmp, t)
        if np.linalg.norm(t_prev-t) <= tol:
            break
        t_prev = copy.deepcopy(t)
    return t, k+1


def matrix_sparser(A, G, c, alpha):
    n = A.shape[0]
    D = np.zeros((n, n))
    z = np.zeros(n)
    e = np.ones(n)
    delta = (1-alpha)/n
    for j in range(n):
        if c[j] != 0:
            D[j][j] = 1 / c[j]
            z[j] = delta
        else:
            z[j] = 1 / n

    return alpha * (G @ D), e * z

def shiftedPowerMethod(A, maxIter=5000, tol=1e-16):
    lambda1, _, _, _ = power_method(A, maxIter=maxIter, tol=tol)
    B = A-abs(lambda1)*np.eye(A.shape[0])
    tmpLambda2, _, _, _ = power_method(B)
    return abs(tmpLambda2)-abs(lambda1)

def shifted_power_method(A, maxIter=5000, tol=1e-16):
    lambda1, v1, _, _ = power_method(A, maxIter=5000, tol=1e-16)
    B = A-abs(lambda1)*np.eye(A.shape[0])
    tmpLambda2, _, _, _ = power_method(B)
    return tmpLambda2

