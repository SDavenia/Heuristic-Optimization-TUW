"""Demo application solving the minimum vertex cover problem.

Given an undirected simple graph, find a minimum subset of the vertices so that from each edge in the graph at
least one of its end points is in this subset.
"""

import random
from typing import Any, Tuple
from itertools import combinations
import heapq
import networkx as nx

from pymhlib.solution import SetSolution, TObj
from pymhlib.settings import get_settings_parser
from pymhlib.scheduler import Result
from pymhlib.demos.graphs import create_or_read_simple_graph

import numpy
import bisect

parser = get_settings_parser()


class SPlexInstance:
    """s-Plex problem instance

    Given an undirected simple graph, find the minimal weight of edited 
    edges to retain a graph of unconnected s-Plexes

    Attributes
        - graph: the graph for which we want to find the s-Plexes
        - s: s-Plex size
        - n: number of nodes
        - m: number of edges
        - weights: nxn matrix filled with weights
        - weights_given_graph: nxn matrix filled weigths of the given graph (added edges have weights, removed edge's weights are 0)
    """

    def __init__(self, filename):
        """Create or read graph with given name."""
        lines = 0
        with open(filename, "r") as infile:
            graph_params = infile.readline().split()
            self.s = int(graph_params[0])
            self.n = int(graph_params[1])
            self.m = int(graph_params[2])
            self.weights = numpy.zeros((self.n + 1, self.n + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
            self.weights_given_graph = numpy.zeros((self.n + 1, self.n + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
            self.neighbors_given_graph = dict.fromkeys(range(1,self.n + 1))
            for k, _ in self.neighbors_given_graph.items(): 
                self.neighbors_given_graph[k] = []
            for line in infile.readlines():
                (a,b,e,w) = [int(x) for x in line.split()]
                self.weights[a][b] = w
                self.weights[b][a] = w
                if e == 1:
                    self.weights_given_graph[a][b] = w
                    self.weights_given_graph[b][a] = w
                    self.neighbors_given_graph[a].append(b)
                    self.neighbors_given_graph[b].append(a)
                    
            print(self.neighbors_given_graph)
            print("given graph weight matrix")
            print(self.weights_given_graph)
            print("weight matrix:")
            print(self.weights)

    """construction heuristic
    
    Attributes:
        - k: a count of nodes to choose from the sorted list of weigts of adjacent edges for the construction
    """
    def construct(self, s, k):
        sorted_nodes = []
        for n, neighbors in enumerate(self.weights_given_graph[1:], start=1):
            sorted_nodes.append([n,numpy.sum(neighbors)])
        list.sort(sorted_nodes, key= lambda x:x[1], reverse= True)

        print("sorted nodes")
        print(sorted_nodes)

        selected_nodes = []
        for i,(node,_) in enumerate(sorted_nodes):
            if [x for x in self.neighbors_given_graph[node] if x in selected_nodes] == []:
                selected_nodes.append(sorted_nodes.pop(i))

                if len(selected_nodes) == k:
                    break

        print("k selected nodes")
        print(selected_nodes)
        print("sorted nodes left")
        print(sorted_nodes)

        clusters = []
        for node in selected_nodes:
            clusters.append([node[0]])

        print("clusters")
        print(clusters)
        while len(sorted_nodes):
            (candidate, _) = sorted_nodes.pop(0) # inefficient, O(n), maybe change from list to deque for O(1) access
            best_cluster = []
            best_weight_to_cluster = 0
            for cluster in clusters:
                print(self.neighbors_given_graph[candidate])
                weight_to_cluster = 0
                for neigbour in [x for x in self.neighbors_given_graph[candidate] if x in cluster]:
                    weight_to_cluster += self.weights[candidate][neigbour]
                print(cluster)
                print(weight_to_cluster)
                if weight_to_cluster > best_weight_to_cluster:
                    best_weight_to_cluster = weight_to_cluster
                    best_cluster = cluster
            best_cluster.append(candidate)
        print(clusters)

if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi.construct(s=1,k=3)
    #run_optimization('Minimum Vertex Cover', VertexCoverInstance, VertexCoverSolution, data_dir + "frb40-19-1.mis")