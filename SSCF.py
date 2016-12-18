from util import *
import numpy as np

A = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]])
P = np.array([
    [0, 1, 1, 2, 3, 3],
    [1, 0, 1, 2, 3, 3],
    [1, 1, 0, 1, 2, 2],
    [2, 2, 1, 0, 1, 1],
    [3, 3, 2, 1, 0, 1],
    [3, 3, 2, 1, 1, 0]])
F = np.array([
    [0.00, 1.00, 0.88, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.88, 0.00, 0.00, 0.00],
    [0.88, 0.88, 0.00, 0.79, 0.00, 0.00],
    [0.00, 0.00, 0.79, 0.00, 0.88, 0.88],
    [0.00, 0.00, 0.00, 0.88, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.88, 1.00, 0.00]])
k = 2
# dataset_path = './data/karate.gml'
# dataset_path = './data/football.gml'
dataset_path = './data/polblogs.zip'
# A = get_adjacent_matrix(dataset_path)
A = get_adjacent_for_pol()
P = find_geodesic_distances(A)
S = find_sim(P)
print(S)
F = find_linear_sparse_code(S)
Es = find_eigen_vectors(F, k)
Ea = find_eigen_vectors(A, k)
centroid, labels, inertia = kmeans(Es, Ea=Ea, n_clusters=2)
print(centroid, labels, inertia)
