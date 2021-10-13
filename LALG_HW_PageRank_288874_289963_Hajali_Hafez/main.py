from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
s = 289963
np.random.seed(s)
path = 'edges_file_python.txt'

def main(path):

    ### Q1
    G, spG, n = adj_mat(path)
    print('The original adjacency matrix:')
    print(G)
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_title('Sparsity pattern of G')
    ax1.spy(spG)
    fig1.savefig('fig1.eps', format='eps')

    ### Q3
    r, c =  in_out_degs(G)
    print('in-degs vector:')
    print(r)
    print('out-degs vector:')
    print(c)

    ### Q5
    M, spM = hyperlink_mat(n, G, c)
    print('The hyperlink matrix:')
    print(M)
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title('Sparsity pattern of M')
    ax2.spy(spM)
    fig2.savefig('fig2.eps', format='eps')

    ### Q7
    M_hat, spM_hat = mod_hyperlink_mat(n, G, c)
    print('The modified hyperlink matrix:')
    print(M_hat)
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    ax3.set_title('Sparsity pattern of M_hat')
    ax3.spy(spM_hat)
    fig3.savefig('fig3.png', format='png')

    ### Q10
    eigsM, vecsM = np.linalg.eig(M_hat)
    lambda1M, v1M, iterM, errM = power_method(M_hat, tol=1e-16, maxIter=5000)
    print('The convergence rate for M_hat power method:', abs(eigsM[1])/abs(eigsM[0]))
    print('The dominant eigenvalue modulus using power method', lambda1M)
    print('The dominant eigenvalue modulus using the direct method (numpy):', abs(eigsM[0]))
    print('The dominant eigenvector using the power method:\n', abs(vecsM[:,0])/np.sum(abs(vecsM[:,0])))
    print('The dominant eigenvector using the direct method (numpy):\n', v1M)
    print('Relative error between eigenvalues:', abs(lambda1M-abs(eigsM[0])/abs(eigsM[0])))
    print('Number of iterations needed for convergence:', iterM)

    ### Q12
    alpha = 0.85
    A = google_mat(G, c, alpha)
    print('The Google Matrix:')
    print(A)

    ### Q13
    converged_exploration, iterConv = explore_web(A, maxIter=5000, tol=1e-16)
    print('The PageRank vector resulting from exploring the web:')
    print(converged_exploration)
    print('The number of iterations needed for exploration to converge:', iterConv)

    ### Q14
    eigsA, vecsA = np.linalg.eig(A)
    lambda1A, v1A, iterA, errA = power_method(A, tol=1e-16, maxIter=5000)
    print('The convergence rate for A power method:', abs(eigsA[1])/abs(eigsA[0]))
    print('The dominant eigenvalue modulus using power method', lambda1A)
    print('The dominant eigenvalue modulus using the direct method (numpy):', abs(eigsA[0]))
    print('The dominant eigenvector using the power method:\n', abs(vecsA[:,0]/np.sum(abs(vecsA[:,0]))))
    print('The dominant eigenvector using the direct method (numpy):\n', v1A)
    print('Relative error between eigenvalues:', abs((lambda1A-abs(eigsA[0]))/abs(eigsA[0])))
    print('Number of iterations needed for convergence:', iterA)
    print('The norm of difference between the actual dominant eigenvector and the approximate one:')
    print(np.linalg.norm(abs(vecsA[:,0]/np.sum(abs(vecsA[:,0])))-v1A.ravel()))
    print('The norm of difference between the actual dominant eigenvector (PageRank vector) and the result of question 13:')
    print(np.linalg.norm(abs(vecsA[:,0]/np.sum(abs(vecsA[:,0])))-converged_exploration.ravel()))

    ### Q15
    print("Google matrix as sum of a sparse matrix and a coefficient matrix")
    print(matrix_sparser(A, G, c, alpha))

    ### Q17
    lambda2M = shifted_power_method(M_hat, maxIter=5000, tol=1e-16)
    print('The second M_hat dominant eigenvalue modulus using power method', lambda2M)
    print('The second M_hat dominant eigenvalue modulus using the direct method (numpy):', abs(eigsM[1]))
    print('Relative error between eigenvalues:', abs((lambda2M-abs(eigsM[1]))/abs(eigsM[1])))

    ### Q19
    page_rank_vector=v1A.ravel()
    fig4, ax = plt.subplots(figsize=(100,10))
    x = np.argsort(page_rank_vector)
    width = 0.35
    sortedVector = np.sort(page_rank_vector)
    ax.bar(x, sortedVector, width=width),
    ax.set_ylim(0,0.05)
    ax.set_xlabel('page index')
    ax.set_ylabel('rank')
    ax.set_title('The page ranks bar graph for all pages')
    fig4.tight_layout()
    fig4.savefig('fig4.eps', format='eps')



    sorted_probs = []
    sorted_ranks_ind = []
    counter = 1
    for i in np.flip(x):
        if counter == 13:
            break
        sorted_ranks_ind.append(counter)
        sorted_probs.append(page_rank_vector[i])
        counter += 1

    fig5, ax = plt.subplots(figsize=(16,9))
    inds = sorted_ranks_ind
    sortedVector = sorted_probs
    ax.bar(inds, sortedVector, width=0.35)
    ax.set_xticks(inds)
    ax.set_ylim(0,0.05)
    ax.set_xlabel('page index')
    ax.set_ylabel('rank')
    ax.set_title('The 12 highest page ranks')
    fig5.tight_layout()
    fig5.savefig('fig5.eps', format='eps')

    indices12 = np.flip(x)[:12]
    outs = c[np.array(indices12)]
    ins = r[np.array(indices12)]
    print('The indices of the highest page ranks')
    print(indices12)
    print('The 12 highest page ranks:')
    print(sortedVector)
    print('The in-degs of the 12 highest page ranks')
    print(ins)
    print('The out-degs of the 12 highest page ranks')
    print(outs)

    plt.show()

if __name__ == '__main__':
    main(path)