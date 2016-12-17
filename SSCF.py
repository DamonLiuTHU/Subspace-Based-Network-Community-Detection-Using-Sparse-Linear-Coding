import numpy as np
from scipy import linalg
import sklearn.cluster as sc
from sklearn import linear_model

def find_sim(P, sigma = 1):
    S = (P*P)/(2 * sigma ** 2)
    return S

def find_linear_sparse_code(S):
    n = np.shape(S)[0]
    F = np.zeros([n, n])
    for i in range(n):
        Sh = np.column_stack((S[:, :i], np.zeros([n, 1]), S[:, i + 1:]))
        lasso = linear_model.Lasso(alpha = 0.05, fit_intercept = False)
        lasso.fit(Sh, S[:, i])
        w = lasso.coef_ / sum(lasso.coef_)
        F[i, :] = F[i, :] + w
    max_dig = []
    for i in range(n):
        col = F[i, :]
        max_dig.append([np.max(col)])
    F = F / max_dig
    F = (F + np.transpose(F))/2
    return F

def find_eigen_vectors(F,k):
    ds = [np.sum(row) for row in F]
    D = np.diag(ds)
    Dn = np.power(np.linalg.matrix_power(D,-1),0.5)
    L = np.identity(len(F)) - (Dn.dot(F)).dot(Dn)

    evals, evcts = linalg.eig(L)
    vals = dict (zip(evals, evcts.transpose()))
    keys = sorted(vals.keys())
    E = np.array([vals[i] for i in keys[1:k]]).transpose()
    return E

def kmeans(Es,Ea = None,n_clusters = 2):
    E = Es
    if Ea is not None:
        E = (np.concatenate((Es.T, Ea.T))).T
    centroid, labels, inertia = sc.k_means(E, n_clusters = n_clusters)
    return centroid, labels, inertia

A = np.array([
    [0,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,0,1,0,0],
    [0,0,1,0,1,1],
    [0,0,0,1,0,1],
    [0,0,0,1,1,0]])
P = np.array([
    [0,1,1,2,3,3],
    [1,0,1,2,3,3],
    [1,1,0,1,2,2],
    [2,2,1,0,1,1],
    [3,3,2,1,0,1],
    [3,3,2,1,1,0]])
F = np.array([
    [0.00,1.00,0.88,0.00,0.00,0.00],
    [1.00,0.00,0.88,0.00,0.00,0.00],
    [0.88,0.88,0.00,0.79,0.00,0.00],
    [0.00,0.00,0.79,0.00,0.88,0.88],
    [0.00,0.00,0.00,0.88,0.00,1.00],
    [0.00,0.00,0.00,0.88,1.00,0.00]])
k = 2

S = find_sim(P)
F = find_linear_sparse_code(S)
Es = find_eigen_vectors(F,k)
Ea = find_eigen_vectors(A,k)
centroid, labels, inertia = kmeans(Es,Ea = Ea, n_clusters = 2)
print(centroid, labels, inertia )


