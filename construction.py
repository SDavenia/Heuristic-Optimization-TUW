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
        
        # To enforce that selected initial nodes should not be neighbours
        selected_nodes = []
        neighbors_selected_nodes = [] # List containing the neighbours of the nodes selected thus far
        for i,(node,_) in enumerate(sorted_nodes):
            if node not in neighbors_selected_nodes:
                selected_nodes.append(sorted_nodes.pop(i))
                neighbors_selected_nodes += self.neighbors_given_graph[node]

                if len(selected_nodes) == k:
                    break
        print(f"Non neighbouring selected nodes: {selected_nodes}")
            
        print("k selected nodes")
        print(selected_nodes)
        print("sorted nodes left")
        print(sorted_nodes)

        ########## ASSIGN THE NODES TO THE CLUSTERS ##########
        clusters = []
        for node in selected_nodes:
            clusters.append([node[0]])
        print(f"Initial clusters:\n{clusters}")

        unassigned = [x[0] for x in sorted_nodes]
        while len(unassigned):
            # Compute similarity of each node to each cluster
            node_cluster_most_similar = [] # Stores for each node what cluster it is more "similar to"
            for node in unassigned:
                node_best_clust = None                  # Stores most similar cluster to node being considered
                node_highest_similarity = float('-inf') # Stores the similarity with the most similar cluster thus far
                
                # Compute the similarity between the node and each cluster
                for ind, clust in enumerate(clusters):
                    node_similarity = 0
                    disjoint = 1   # If node is disjoint from cluster we want dist to it to remain -inf

                    for elem in clust:
                        if self.weights_given_graph[node, elem] != 0: 
                            node_similarity += self.weights_given_graph[node, elem] 
                            disjoint = 0 # If at least one edge was found it means it is not disjoint from clust
                        else:
                            node_similarity -= self.weights[node, elem]
                    
                    if node_similarity > node_highest_similarity and disjoint == 0:
                        node_highest_similarity = node_similarity
                        node_best_clust = ind

                # Store to what cluster it is more similar to
                node_cluster_most_similar.append([node, node_best_clust, node_highest_similarity])
            
            # Extract node with maximum similarity to a cluster & Assign it
            candidate = max(node_cluster_most_similar, key=lambda x:x[2]) 
            if candidate[1] is None: # Disjoint node
                clusters.append([candidate[0]])
            else:
                clusters[candidate[1]].append(candidate[0])

            print(f"New clusters: {clusters}")
            unassigned.remove(candidate[0])

        ########## TURN THE CLUSTERS INTO S-PLEXES ##########
        # We have to count the number of edges within the s-plex
        for clust in clusters:
            n_nodes = len(clust)
            cluster_neighbours = {node:[] for node in clust}
            for node in clust:
                cluster_neighbours[node] = [x for x in self.neighbors_given_graph[node] if x in clust]
            print(f"Cluster {clust}:\nNeighbours list {cluster_neighbours}")

            # Now we need the order of each node in the subgraph
            count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
            print(f"Number of neighbours for each node is {count_neighbours}")

            # List of nodes which do not have order for the s-plex assumption
            nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
            print(f"Nodes which do not satisfy are {nodes_not_satisfied}")
            
            # Now we create a list of potential edges to add where we only consider pairs where at least one of the nodes is a unsatisfactory one
            # Now we add edges with minimum cost at every iteration
            potential_edges = []
            # This is quite inefficient as more checks than necessary
            for ind, node_i in enumerate(clust):
                for node_j in clust[ind+1 : ]:
                    if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied:
                        if self.weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                            potential_edges.append([[node_i, node_j],self.weights[node_i, node_j]]) # [[node_i, node_j], weight]
            potential_edges.sort(key=lambda x:x[1]) # Sort in decreasing order
            print(f"Potential edges: {potential_edges}")

            while nodes_not_satisfied:
                candidate_edge = potential_edges.pop(0)
                node_i = candidate_edge[0][0]
                node_j = candidate_edge[0][1]
                cluster_neighbours[node_i].append(node_j)
                cluster_neighbours[node_j].append(node_i)
                count_neighbours[node_i] += 1
                count_neighbours[node_j] += 1
                nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                print(f"Adding edge between ({node_i}, {node_j})")
                print(f"Nodes which do not satisfy are {nodes_not_satisfied}")

        
        # Return a solution




if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi.construct(s=1,k=3)
    #run_optimization('Minimum Vertex Cover', VertexCoverInstance, VertexCoverSolution, data_dir + "frb40-19-1.mis")