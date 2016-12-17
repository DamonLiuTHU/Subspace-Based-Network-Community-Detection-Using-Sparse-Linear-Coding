import numpy
# from igraph import *
import networkx as nx

def get_data_from_file(path):
    graph = nx.read_gml(path)
    return graph


def get_adjacent_matrix(path):
    matrix = nx.to_numpy_matrix(get_data_from_file(path))
    return matrix
