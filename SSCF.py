from util import *
import numpy as np
import math
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
dataset_path = './data/karate.gml'
# dataset_path = './data/football.gml'
A = get_adjacent_matrix(dataset_path)
# A = get_adjacent_for_pol()
print('A')
print(type(A))
A = np.array(A)

P = find_geodesic_distances(A)

print('P')
print(P)
S = find_sim(P)
print('S')

print(S)
F = find_linear_sparse_code(S)
print('F')
print(F)
Es = find_eigen_vectors(F,k)
Ea = find_eigen_vectors(A,k)
centroid, labels, inertia, best_n_iter = kmeans(Es,Ea = Ea, n_clusters = 2)
print('-----------------------')
print(centroid, labels, inertia, best_n_iter )


