import numpy
from igraph import *

def get_data_from_file(path):
    g = Graph.Read_GML(path)
    return g

def get_adjacent_matrix(path):
    graph = get_data_from_file(path)
    tmp = numpy.mat(graph.get_adjacency())
    print(type(tmp))
    return tmp

    # tmp = get_adjacent_matrix('./data/football.gml')
    # print tmp
