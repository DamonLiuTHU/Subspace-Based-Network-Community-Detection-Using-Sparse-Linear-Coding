import Tools
import math
import numpy


# from geographiclib.geodesic import Geodesic


# dataset is A
def sparse_subspace_communities_with_fusion(dataset_path, sigma):
    matrix = Tools.get_adjacent_matrix(dataset_path)
    P = Find_Geodesic_Distances(matrix)
    P = P * P
    S = -P / (2 * sigma ** 2)

    S = S.getA()
    width = len(S)
    for i in range(width):
        for j in range(width):
            # tmp = S[i][j]
            S[i][j] = math.exp(S[i][j])
    S = numpy.mat(S)
    return S


def Find_Geodesic_Distances(M):
    D = []
    n = len(M)
    D.append(M)
    for k in range(1, n + 1):
        s = (n, n)
        current = numpy.mat(numpy.zeros(s))
        D.append(current)
        previous = D[k - 1]

        previous = previous.getA()
        current = current.getA()

        for i in range(n):
            for j in range(n):
                if previous[i][j] > 0: current[i][j] = previous[i][j]
                if i == j: continue
                value1 = previous[i][k - 1]  # from i to k
                value2 = previous[k - 1][j]  # from k to j

                if value1 != 0 and value2 != 0:  # if there is an edge from i to k and k to j then there is edge from i to j
                    if current[i][j] != 0:
                        current[i][j] = min(current[i][j], value2 + value1)
                    else:  # this means previously there is no edge from i to j
                        current[i][j] = value2 + value1

        D[len(D) - 1] = numpy.mat(current)
    print "APSP (k=%d):" % k
    print_graph(D[len(D) - 1])
    return D[len(D) - 1]


def print_graph(G):
    for row in G:
        print row


sparse_subspace_communities_with_fusion('./data/football.gml', 10)
