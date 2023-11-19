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
        - sorted_edges: tuples (w,(a,b)) with weights and nodes, sorted by decreasing weights
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
            self.sorted_edges = []
            self.edges_in_graph = []
            for line in infile.readlines():
                (a,b,e,w) = [int(x) for x in line.split()]
                self.weights[a][b] = w
                self.weights[b][a] = w
                if e == 1:
                    self.sorted_edges.append((w,(a,b)))
                    self.edges_in_graph.append((a,b))


            list.sort(self.sorted_edges, key= lambda x:x[0], reverse= True)

    def construct(self):
        node_marker = numpy.zeros(self.n + 1) # add 1 because input file starts nodes with 1 (not 0)
        components = []
        for (w,(a,b)) in self.sorted_edges:
            if (node_marker[a] == 0 and node_marker[b] == 0):
                components.append(((a,b),w))
                node_marker[a] = 1
                node_marker[b] = 1
        for i in range(1, len(node_marker)):
            if not node_marker[i]:
                components.append(([i],0))
                node_marker[i] = 1
        
        weights_temp = numpy.zeros((len(components) + 1, len(components) + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
        print(components)
        for i,(nodesA,_) in enumerate(components,1):
            for j,(nodesB,_) in enumerate(components[i:],i+1):
                for nodeA in nodesA:
                    for nodeB in nodesB:
                        if tuple(sorted((nodeA,nodeB))) in self.edges_in_graph:
                            weights_temp[i][j] = weights_temp[i][j] + self.weights[nodeA][nodeB]
                weights_temp[j][i] = weights_temp[i][j]



        print(weights_temp)

if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi.construct()
    #run_optimization('Minimum Vertex Cover', VertexCoverInstance, VertexCoverSolution, data_dir + "frb40-19-1.mis")