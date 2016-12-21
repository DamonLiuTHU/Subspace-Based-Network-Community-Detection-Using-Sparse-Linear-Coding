from util import *
import numpy as np

'''
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
'''

# dataset karate
# dataset_path = './data/karate.gml'
# A = get_adjacent_matrix(dataset_path)
# real_label,dic = get_karate_label()

# dataset football
# dataset_path = './data/football.gml'
# A = get_adjacent_matrix(dataset_path)
# real_label,dic = get_football_label()

# dataset pol
A = get_adjacent_for_pol()
real_label,dic = get_pol_label()
print('A')
A = np.array(A)
P = find_geodesic_distances(A)
print('P')
S = find_sim(P)
print('S')
F = find_linear_sparse_code(S)
print('F')
evals_s, evcts_s = find_eigen_vectors(F)
evals_a, evcts_a = find_eigen_vectors(A)
print('get all eigen_vectors')

def k_means_2(k=2):
    Es = find_k_eigen_vetors(evals_s, evcts_s, k)
    Ea = find_k_eigen_vetors(evals_a, evcts_a, k)
    centroid, labels, inertia, best_n_iter,nmi = kmeans(Es, real_label, Ea = Ea, n_clusters = k)
    norm_error_2 = inertia ** 0.5 / k
    return norm_error_2
def k_means_ite(k,norm_error_2):
    Es = find_k_eigen_vetors(evals_s, evcts_s, k)
    Ea = find_k_eigen_vetors(evals_a, evcts_a, k)
    centroid, labels, inertia, best_n_iter,nmi = kmeans(Es, real_label, Ea = Ea, n_clusters = k)
    normlized_error = inertia ** 0.5 / k
    normlized_error = normlized_error / norm_error_2
    print('-----------------------',k)
    print( labels, normlized_error, best_n_iter )
    print(real_label)
    print(nmi)
    return labels, nmi, normlized_error, best_n_iter
norm_error_2 = k_means_2()
for k in range(2,16):
    k_means_ite(k,norm_error_2)