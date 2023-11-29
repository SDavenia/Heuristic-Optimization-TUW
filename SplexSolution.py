from pymhlib.solution import Solution, TObj
from pymhlib.settings import get_settings_parser
from pymhlib.scheduler import Result
import time
from SPlexInstance import SPlexInstance

from operator import itemgetter
import random
import numpy

from copy import deepcopy as dcopy
from utilities import Utilities

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class SPlexSolution(Solution):
    """
    A solution of the s-plex editing problem.
    It contains the following attributes
        - s: plex order desired
        - weights: instance weights
        - weights_given_graph: weights of edges already present.
        - initial_neighbors: initial neighbourhood list for the nodes
        - current_neighbours: neighborhood list given the current solution
        - clusters: set of clusters of the solution.
        - edges_modified: solution representation, contains what edges were removed/added from the initial list
    """
    to_maximise = False
    
    def __init__(self, inst: SPlexInstance = None):
        super().__init__()
        self.inst = inst
        self.s = inst.s
        self.weights = inst.weights
        self.weights_given_graph = inst.weights_given_graph
        self.initial_neighbors = inst.neighbors_given_graph
        self.current_neighbours = dcopy(self.initial_neighbors)

        # edges_modified contains the solution which will be written to file.
        # clusters is used to help as it is often used for the neighbourhood structures as well.
        self.clusters = []
        self.edges_modified = [] # Actual solution to the problem.

        # Needed for writing to file
        self.problem_instance = inst.problem_instance

    def calc_objective(self)->int:
        cost = 0
        for edge in self.edges_modified:
            cost += self.weights[edge[0], edge[1]]
        # print(f"The given solution has cost: {cost}")
        return cost
    
    def __repr__(self):
        return f"Solution edges: {self.edges_modified}"
    
    def update_current_neighbours(self):
        """
        It updates the values of current_neighbours given the current solution.
        """
        self.current_neighbours = dcopy(self.initial_neighbors)
        #print(f"We applied these modifications in our solution {self.edges_modified}")
        for edge in self.edges_modified:
            if edge[1] in self.initial_neighbors[edge[0]]:  #The edge was present initially, so we removed it
                self.current_neighbours[edge[1]].remove(edge[0])
                self.current_neighbours[edge[0]].remove(edge[1])
            else:                                           #The edge was NOT present initially, so we added it
                self.current_neighbours[edge[1]].append(edge[0])
                self.current_neighbours[edge[0]].append(edge[1])


    def check(self):
        # We check if all our clusters are s-plexes and if there are no edges between them
        self.update_current_neighbours() # Ensure we are on the right updates
        for clust in self.clusters:
            # First of all we check if there are external edges (to the cluster)
            for element in clust:
                element_neighbours = self.current_neighbours[element]
                if any([x not in clust for x in element_neighbours]): # means that there is an edge to a non-cluster element
                    return False

            # Now we compute the order of each node in the cluster and compare it
            n_clust = len(clust)
            counts = [len(value) for key,value in self.current_neighbours.items() if key in clust]
            #print(f"I have these counts: {counts}")
            if any([x < n_clust - self.s for x in counts]):
                return False
        #print(f"VALID SOLUTION")
        return True

    def copy(self):
        sol = SPlexSolution(self.inst)
        sol.copy_from(self)
        return sol
    
    def copy_from(self, other: "SPlexSolution"):
        super().copy_from(other)
        self.clusters = dcopy(other.clusters)
        self.current_neighbours = dcopy(other.current_neighbours)
        self.edges_modified = dcopy(other.edges_modified)
        self.s = dcopy(other.s)
        self.initial_neighbors = dcopy(other.initial_neighbors)
        self.weights = dcopy(other.weights)
        self.weights_given_graph = dcopy(other.weights_given_graph)
    
    def initialize(self):
        """
        Reset the solution
        """
        self.clusters = []
        self.edges_modified = []
        self.update_current_neighbours()
    
    def construct_set_initial(self, k, alpha):
        """
        Selects the initial k nodes for the clusters with a level of randomization alpha
        Returns sorted_nodes which is required for the construction.
        """        

        # Obtain a list of nodes with weight of their edges, and sort in decreasing order
        sorted_nodes = []
        for n, neighbors in enumerate(self.weights_given_graph[1:], start=1):
            if numpy.sum(neighbors) == 0:  # Identify disjoint nodes and leave them untouched since they are already an s-plex
                next
            sorted_nodes.append([n,numpy.sum(neighbors)])
        list.sort(sorted_nodes, key= lambda x:x[1], reverse= True)
        #print(f"Sorted nodes:\n{sorted_nodes}")

        # Select k nodes for initial clusters enforcing that they should not be neighbours
        selected_nodes = []
        neighbors_selected_nodes = [] # List containing the neighbours of the nodes selected thus far

        while(len(selected_nodes) != k):
            CL = [x for x in sorted_nodes if x[0] not in neighbors_selected_nodes]
            if not CL:
                break
            cmax = CL[0][1]
            cmin = CL[-1][1]
            threshold = cmin + alpha * (cmax - cmin)
            RCL = [x[0] for x in sorted_nodes if x[1] >= threshold and x[0] not in neighbors_selected_nodes] # Only keep the nodes

            # Necessary to make it deterministic
            if alpha == 1:
                selected_node = RCL.pop(0)
            else:
                selected_node = RCL.pop(random.randint(0, len(RCL)-1)) # Choose at random a node from the candidate list.
            sorted_nodes = [x for x in sorted_nodes if x[0] != selected_node]
            selected_nodes.append(selected_node)
            neighbors_selected_nodes += self.initial_neighbors[selected_node] # Add neighours of that node to it while also capping the first 50

        #print(f"Selected nodes: {selected_nodes}")
        # Initialize the clusters
        for node in selected_nodes:
            self.clusters.append([node])
        return sorted_nodes
    
    def construct_assign_nodes(self, k, beta, unassigned_nodes):
        node_similarity_to_cluster = {x[0]:[float('-inf')] * k for x in unassigned_nodes} # Dictionary with node: (dist_clust_1, ..., dist_clust_k)
        
        # Compute initial similarity to each cluster for each node
        for node in node_similarity_to_cluster.keys():
            node_similarity_to_cluster[node] = [self.weights[node, clust[0]] if self.weights_given_graph[node, clust[0]] != 0 else -self.weights[node, clust[0]] for clust in self.clusters]
        #print(f"Initial nodes smilarities to clusters:\n\t{node_similarity_to_cluster}")
        # Now assign nodes based on similarity
        """
        THIS ONE CONSIDERS BEST ASSIGNMENT FOR EACH NODE and chooses among those above threshold at random
        VERSION BELOW (implemented one) considers all possible node, cluster pairs and selects only those above a threshold at random

        while node_similarity_to_cluster:
            # CL = node_similarity_to_cluster
            node_max = {max(value) for value in node_similarity_to_cluster.values()} # Highest similarity for each node
            cmax = max(node_max) 
            cmin = min(node_max)
            threshold = cmin + beta*(cmax - cmin)

            RCL = [x for x in node_similarity_to_cluster if max(node_similarity_to_cluster[x]) >= threshold]
            if beta == 1:
                node_assigned = max(node_similarity_to_cluster, key=lambda k: max(node_similarity_to_cluster[k]))  # Node with highest similarity chosen deterministically (first in list)
            else:
                node_assigned = RCL.pop(random.randint(0, len(RCL)-1))

            # Extract to what cluster we should assign it to            
            cluster_assigned, _ = max(enumerate(node_similarity_to_cluster[node_assigned]), key=itemgetter(1))
            self.clusters[cluster_assigned].append(node_assigned)

            del node_similarity_to_cluster[node_assigned]
            for node in node_similarity_to_cluster.keys():
                if self.weights_given_graph[node, node_assigned] != 0:
                    node_similarity_to_cluster[node][cluster_assigned] += self.weights[node, node_assigned]
                else:
                    node_similarity_to_cluster[node][cluster_assigned] -= self.weights[node, node_assigned]
        """

        all_ranks = []
        while node_similarity_to_cluster:
            node_max = {max(value) for value in node_similarity_to_cluster.values()} # Highest similarity for each node
            node_min = {min(value) for value in node_similarity_to_cluster.values()}
            cmax = max(node_max) 
            cmin = min(node_min)
            #print(f"Cmax: {cmax}, Cmin: {cmin}")
            threshold = numpy.add(cmin, numpy.multiply(beta, numpy.subtract(cmax,cmin)))


            # Consider all pairings of node-cluster which are above the threshold
            RCL_pairs = [(key, element) for key, values in node_similarity_to_cluster.items() for element in values if element >= threshold]
            #print(f"Result threshold {threshold} pairs: {RCL_pairs}")
            if beta == 1:
                node_assigned = max(node_similarity_to_cluster, key=lambda k: max(node_similarity_to_cluster[k]))  # Node with highest similarity chosen deterministically (first in list)
                best_clusters = sorted(enumerate(node_similarity_to_cluster[node_assigned]), key=itemgetter(1), reverse=True)
                #all_ranks += [(x[0], x[1], node_assigned) for x in best_clusters]
                all_ranks.append((node_assigned, best_clusters[0][0]))
            else:
                pair_assigned = RCL_pairs.pop(random.randint(0, len(RCL_pairs)-1))
                node_assigned = pair_assigned[0]
                #print(f"NODE {node_similarity_to_cluster[node_assigned]}")
                cluster_assigned = [index for index, value in enumerate(node_similarity_to_cluster[node_assigned]) if value == pair_assigned[1]][0] #[0] To remove duplicates if present

            if beta == 1:
                del node_similarity_to_cluster[node_assigned]

            else:
                #print(f"Assigning node {node_assigned} to cluster {cluster_assigned}")
                self.clusters[cluster_assigned].append(node_assigned)
                del node_similarity_to_cluster[node_assigned]
                for node in node_similarity_to_cluster.keys():
                    if self.weights_given_graph[node, node_assigned] != 0:
                        node_similarity_to_cluster[node][cluster_assigned] = numpy.add(node_similarity_to_cluster[node][cluster_assigned], self.weights[node, node_assigned])
                    else:
                        node_similarity_to_cluster[node][cluster_assigned] = numpy.subtract(node_similarity_to_cluster[node][cluster_assigned], self.weights[node, node_assigned])
        if beta == 1:
            for node, cluster in all_ranks:
                self.clusters[cluster].append(node)

        #print(f"Final clusters:\n\t{self.clusters}")

    def construct_all_splex(self):
        """
        Construct s-plex for all the clusters
        """
        for clust in self.clusters:
            self.construct_splex(clust)
        
        # Remove duplicates
        self.edges_modified = list(set(map(frozenset, self.edges_modified)))
        self.edges_modified = [sorted(list(fs)) for fs in self.edges_modified]
        #print(f"Solution edges: {self.edges_modified}")

    def construct_splex(self, clust):
        """
        Construct splex for the given cluster
        """
        for node in clust:
            non_cluster_neighbours = [x for x in self.initial_neighbors[node] if x not in clust]
            # print(f"Node is {node} and it has non cluster neighbours:\n\t{non_cluster_neighbours}")
            # Add to solution the edges between clusters (which have been removed)
            if non_cluster_neighbours:
                #print("THERE ARE NON CLUSTER NEIGHBOURS")
                edges_removed = [set([node, x]) for x in non_cluster_neighbours]
                #print(f"For node {node} we removed edges {edges_removed}")
                self.edges_modified += edges_removed

        # Now we ensure that we make the cluster into a desired s-plex
        n_nodes = len(clust)
        cluster_neighbours = {node:[] for node in clust}
        for node in clust:
            cluster_neighbours[node] = [x for x in self.initial_neighbors[node] if x in clust]
        #print(f"Cluster {clust}:\nNeighbours list {cluster_neighbours}")

        # Now we need the order of each node in the subgraph
        count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
        #print(f"Number of neighbours for each node is {count_neighbours}")

        # List of nodes which do not have order for the s-plex assumption
        nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
        #print(f"Nodes which do not satisfy are {nodes_not_satisfied}")
        
        # Now we create a list of potential edges to add where we only consider pairs where at least one of the nodes is a unsatisfactory one
        # Now we add edges with minimum cost at every iteration
        # print(f"Cluster:\n\t{clust}")
        if len(nodes_not_satisfied) != 0:
            potential_edges = []
            # This is quite inefficient as more checks than necessary
            for ind, node_i in enumerate(clust):
                for node_j in clust[ind+1 : ]:
                    if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied: # only consider edges between unsatisfied nodes.
                        if self.weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                            potential_edges.append([[node_i, node_j],self.weights[node_i, node_j]]) # [[node_i, node_j], weight]
            potential_edges.sort(key=lambda x:x[1]) # Sort in decreasing order
            #print(f"Potential edges: {potential_edges}")
            #print(f"Nodes not satisfied:\n\t{nodes_not_satisfied}")
            #print(f"Count of neighbours:\n\t{count_neighbours}\nAnd we need {n_nodes} - {self.s}")

            while nodes_not_satisfied:
                candidate_edge = potential_edges.pop(0)
                node_i = candidate_edge[0][0]
                node_j = candidate_edge[0][1]
                cluster_neighbours[node_i].append(node_j)
                cluster_neighbours[node_j].append(node_i)
                count_neighbours[node_i] += 1
                count_neighbours[node_j] += 1
                nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                #print(f"Adding edge between ({node_i}, {node_j})")
                self.edges_modified.append(set([node_i, node_j]))  # Append additional edges we inserted
                edge = [node_i, node_j] if node_i < node_j else [node_j, node_i]
                #print(f"Nodes which do not satisfy are {nodes_not_satisfied}")
                
                # Have to update potential edges to only consider the nodes which are yet not satisfied
                potential_edges = [item for item in potential_edges if item[0][0] in nodes_not_satisfied or item[0][1] in nodes_not_satisfied]
                # print(f"Potential edges: {potential_edges}")

    


    def construct_randomized(self, k, alpha= 1, beta= 1):
        """ 
        Construction algorithm to build a solution:
            - k: number of clusters to obtain
            - alpha: specifies degree of randomization in selection of initial nodes.
            - beta: specifies degree of randomization when selecting next node to add to cluster
        """
        # Select initial clusters and extract unassigned nodes
        unassigned_nodes = self.construct_set_initial(k, alpha)
        # Assign all nodes to some cluster
        self.construct_assign_nodes(k, beta, unassigned_nodes)
        # Convert the decided clusters into an s-plex        
        self.construct_all_splex()
        # self.update_current_neighbours() # Called to update current neighbours as well given the solution found so far
    
    def construct_deterministic(self, k):
        return self.construct_randomized(k, alpha=1, beta=1) # or something similar where alpha is a probabilistic parameter

    def ls_move1node_faster(self, step_function = "best") -> bool:
        """
        Performs one iteration of local search using the moving of one node from one cluster to another
        Returns True if an improevd solution is found.
        Should be faster than the previous implementation
        """
        best_sol = self.copy()
        best_sol_value = self.calc_objective()
        better_found = False
        
        if step_function in ['best', 'first']:
            time1 = time.time()
            initial_clusters = dcopy(self.clusters)
            initial_edges_modified = dcopy(self.edges_modified)
            initial_value = self.calc_objective() # Needed for how delta is evaluated and how we move through
            # print(f"Initialization took {time.time() - time1}s")
            
            for node in range(1, len(self.weights)):
                # time2 = time.time()
                delta_baseline = initial_value
                self.clusters = dcopy(initial_clusters)
                self.edges_modified = dcopy(initial_edges_modified)
                initial_clust = [index for index, sublist in enumerate(self.clusters) if node in sublist][0] # Find what cluster node belongs to 
                current_clust = initial_clust
                # print(f"First deep copies took {time.time() - time2}s")

                for dest_clust in range(len(self.clusters)): # index for which cluster we are working on
                    if dest_clust != initial_clust:
                        # time3 = time.time()
                        delta = 0 # To store changes made by the local move
                        self.clusters[current_clust].remove(node)

                        # time4 = time.time()
                        # Remove from solution all edges between node and elements in old cluster which were added to make s-plex
                        for old_node in self.clusters[current_clust]:
                            edge = [node, old_node] if node < old_node else [old_node, node]
                            if edge in self.edges_modified:
                                delta -= self.weights[edge[0], edge[1]]
                                self.edges_modified.remove(edge)

                            # Add to solution all edges which were in the original graph between node and old elements
                            elif old_node in self.initial_neighbors[node]:
                                edge = [node, old_node] if node < old_node else [old_node, node]
                                delta += self.weights[edge[0], edge[1]]
                                self.edges_modified.append(edge)
                        # print(f"Preliminary manipulation 1 took {time.time() - time4}")


                        # time5 = time.time()
                        # Remove from solution all edges between node and elements in new cluster which were added to isolate it
                        for new_node in self.clusters[dest_clust]:
                            edge = [node, new_node] if node < new_node else [new_node, node]
                            if edge in self.edges_modified:
                                delta -= self.weights[edge[0], edge[1]]
                                self.edges_modified.remove(edge)
                        # print(f"Preliminary manipulation 2 took {time.time() - time5}")
                        
                        # time6 = time.time()
                        # Remove from solution all edges between nodes in the new cluster which were added to make s-plex as we have to rebuild it
                        for ind1, new_node1 in enumerate(self.clusters[dest_clust]):
                            for ind2, new_node2 in enumerate(self.clusters[dest_clust]):
                                edge = [new_node1, new_node2] if new_node1 < new_node2 else [new_node2, new_node1]
                                if edge in self.edges_modified:
                                    delta -= self.weights[edge[0], edge[1]]
                                    self.edges_modified.remove(edge)
                        # print(f"Preliminary manipulation 3 took {time.time() - time6}")
                        # print(f"Preliminary manipulations took {time.time() - time3}")
                        
                        # time7 = time.time()
                        # Append new node
                        self.clusters[dest_clust].append(node)
                        current_clust = dest_clust
                        # self.update_current_neighbours()
                        #print(f"Now we have clusters:\n\t{self.clusters}")
                        #print(f"We start with these edges:\n\t{self.edges_modified}")
                        
                        # Now we rebuild the s-plex for that cluster by only adding edges within
                        n_nodes = len(self.clusters[dest_clust])
                        cluster_neighbours = {x:[] for x in self.clusters[dest_clust]}

                        # Build a dictionary with node and its cluster neighbours
                        #print(f"NEW CLUSTER: {self.clusters[dest_clust]}")
                        #print(f"Initial neighbours: {self.initial_neighbors}")
                        for clust_node in self.clusters[dest_clust]: 
                            cluster_neighbours[clust_node] = [x for x in self.initial_neighbors[clust_node] if x in self.clusters[dest_clust]]
                        #print(f"Cluster {self.clusters[dest_clust]}:\nNeighbours list {cluster_neighbours}")
                        
                        # Count number of neighbours for each
                        count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
                        nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                        #print(f"Number of neighbours for each node is {count_neighbours}")
                        #print(f"Neighbours not satisfying assumption are {nodes_not_satisfied}")
                        
                        # Consider list of potential edges to add
                        if len(nodes_not_satisfied) != 0:
                            potential_edges = []
                            # This is quite inefficient as more checks than necessary
                            for ind, node_i in enumerate(self.clusters[dest_clust]):
                                for node_j in self.clusters[dest_clust][ind+1 : ]:
                                    if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied: # only consider edges between unsatisfied nodes.
                                        if self.weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                                            edge_to_append = [node_i, node_j] if node_i < node_j else [node_j, node_i]
                                            potential_edges.append([edge_to_append ,self.weights[node_i, node_j]]) # [[node_i, node_j], weight]
                            potential_edges.sort(key=lambda x:x[1]) # Sort in decreasing order
                            # print(f"Potential edges: {potential_edges}")

                            while nodes_not_satisfied:
                                candidate_edge = potential_edges.pop(0)
                                node_i = candidate_edge[0][0]
                                node_j = candidate_edge[0][1]
                                cluster_neighbours[node_i].append(node_j)
                                cluster_neighbours[node_j].append(node_i)
                                count_neighbours[node_i] += 1
                                count_neighbours[node_j] += 1
                                nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                                #print(f"Adding edge between ({node_i}, {node_j})")
                                self.edges_modified.append(candidate_edge[0])  # Append additional edges we inserted
                                delta += self.weights[node_i, node_j]
                                # Have to update potential edges to only consider the nodes which are yet not satisfied
                                potential_edges = [item for item in potential_edges if item[0][0] in nodes_not_satisfied or item[0][1] in nodes_not_satisfied]

                        # print(f"Building the s-plex took {time.time() - time7}")
                        # return 
                        #print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")
                        #print(f"Computed with delta evaluation our solution has value: {delta_baseline + delta}")
                        #t1 = time.time()
                        #self.update_current_neighbours() # DONT THINK THIS IS NEEDED
                        # print(f"Update takes: {time.time() - t1}")

                        if delta + delta_baseline < best_sol_value:
                            # Means we found a better solution
                            better_found = True
                            best_sol = self.copy()
                            best_sol_value = delta + delta_baseline
                            if step_function == 'first':
                                return better_found
     
                        delta_baseline = delta_baseline + delta

            self.copy_from(best_sol)
            return better_found
        elif step_function == 'random':
            delta = 0
            
            # Pick a node at random and move it to another cluster at random and see if it improved
            node = random.randint(1, len([x for clust in self.clusters for x in clust]))
            dest_clust = random.randint(0, len(self.clusters)-1)
            initial_cluster = [index for index, sublist in enumerate(self.clusters) if node in sublist][0]
            current_clust = initial_cluster
            while dest_clust == initial_cluster: # Generate until you obtain a new cluster diff than initial one
                dest_clust = random.randint(0, len(self.clusters)-1)
            
            self.clusters[current_clust].remove(node)
            # Remove from solution all edges between node and elements in old cluster which were added to make s-plex
            for old_node in self.clusters[current_clust]:
                edge = [node, old_node] if node < old_node else [old_node, node]
                if edge in self.edges_modified:
                    self.edges_modified.remove(edge)
                    delta = delta - self.weights[edge[0], edge[1]]

                # Add to solution all edges which were in the original graph between node and old elements
                elif old_node in self.initial_neighbors[node]:
                    edge = [node, old_node] if node < old_node else [old_node, node]
                    self.edges_modified.append(edge)
                    delta = delta + self.weights[edge[0], edge[1]]

            # Remove from solution all edges between node and elements in new cluster which were added to isolate it
            for new_node in self.clusters[dest_clust]:
                edge = [node, new_node] if node < new_node else [new_node, node]
                if edge in self.edges_modified:
                    self.edges_modified.remove(edge)
                    delta = delta - self.weights[edge[0], edge[1]]
            
            # Remove from solution all edges between nodes in the new cluster which were added to make s-plex as we have to rebuild it
            for ind1, new_node1 in enumerate(self.clusters[dest_clust]):
                for ind2, new_node2 in enumerate(self.clusters[dest_clust]):
                    edge = [new_node1, new_node2] if new_node1 < new_node2 else [new_node2, new_node1]
                    if edge in self.edges_modified:
                        self.edges_modified.remove(edge)
                        delta = delta - self.weights[edge[0], edge[1]]

            # Append new node
            self.clusters[dest_clust].append(node)
            current_clust = dest_clust
            # self.update_current_neighbours()
            # print(f"RANDOM: Now we have clusters:\n\t{self.clusters}")
            # print(f"We start with these edges:\n\t{self.edges_modified}")
            
            # Now we rebuild the s-plex for that cluster by only adding edges within
            n_nodes = len(self.clusters[dest_clust])
            cluster_neighbours = {x:[] for x in self.clusters[dest_clust]}

            # Build a dictionary with node and its cluster neighbours
            #print(f"NEW CLUSTER: {self.clusters[dest_clust]}")
            #print(f"Initial neighbours: {self.initial_neighbors}")
            for clust_node in self.clusters[dest_clust]: 
                cluster_neighbours[clust_node] = [x for x in self.initial_neighbors[clust_node] if x in self.clusters[dest_clust]]
            #print(f"Cluster {self.clusters[dest_clust]}:\nNeighbours list {cluster_neighbours}")
            
            # Count number of neighbours for each
            count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
            nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
            #print(f"Number of neighbours for each node is {count_neighbours}")
            #print(f"Neighbours not satisfying assumption are {nodes_not_satisfied}")
            
            # Consider list of potential edges to add
            if len(nodes_not_satisfied) != 0:
                potential_edges = []
                # This is quite inefficient as more checks than necessary
                for ind, node_i in enumerate(self.clusters[dest_clust]):
                    for node_j in self.clusters[dest_clust][ind+1 : ]:
                        if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied: # only consider edges between unsatisfied nodes.
                            if self.weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                                edge_to_append = [node_i, node_j] if node_i < node_j else [node_j, node_i]
                                potential_edges.append([edge_to_append ,self.weights[node_i, node_j]]) # [[node_i, node_j], weight]
                potential_edges.sort(key=lambda x:x[1]) # Sort in decreasing order
                # print(f"Potential edges: {potential_edges}")

                while nodes_not_satisfied:
                    candidate_edge = potential_edges.pop(0)
                    node_i = candidate_edge[0][0]
                    node_j = candidate_edge[0][1]
                    cluster_neighbours[node_i].append(node_j)
                    cluster_neighbours[node_j].append(node_i)
                    count_neighbours[node_i] += 1
                    count_neighbours[node_j] += 1
                    nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                    #print(f"Adding edge between ({node_i}, {node_j})")
                    self.edges_modified.append(candidate_edge[0])  # Append additional edges we inserted
                    delta = delta + self.weights[candidate_edge[0][0], candidate_edge[0][1]]
                    # Have to update potential edges to only consider the nodes which are yet not satisfied
                    potential_edges = [item for item in potential_edges if item[0][0] in nodes_not_satisfied or item[0][1] in nodes_not_satisfied]


            if delta < 0:
                best_sol = self.copy()
                return True
            else:
                self.copy_from(best_sol)
                return False


    def ls_move1node_simplified(self, step_function = "best") -> bool:
        """
        Performs one iteration of local search using the moving of one node from one cluster to another
        Returns True if an improevd solution is found.
        Instead of re-building the s-plex we move to it only ensures that the node is made into an s-plex for the cluster
        """
        best_sol = self.copy()
        best_sol_value = self.calc_objective()
        better_found = False
        
        if step_function in ['best', 'first']:
            initial_clusters = dcopy(self.clusters)
            initial_edges_modified = dcopy(self.edges_modified)
            initial_value = self.calc_objective() # Needed for how delta is evaluated and how we move through
            
            for node in range(1, len(self.weights)):
                # time2 = time.time()
                delta_baseline = initial_value
                self.clusters = dcopy(initial_clusters)
                self.edges_modified = dcopy(initial_edges_modified) # Moved at the end for small efficiency gain
                initial_clust = [index for index, sublist in enumerate(self.clusters) if node in sublist][0] # Find what cluster node belongs to 
                current_clust = initial_clust
                #print(f"First deep copies took {time.time() - time2}s")

                for dest_clust in range(len(self.clusters)): # index for which cluster we are working on
                    if dest_clust != initial_clust:
                        # time3 = time.time()
                        delta = 0 # To store changes made by the local move
                        self.clusters[current_clust].remove(node)

                        # time4 = time.time()
                        # Remove from solution all edges between node and elements in old cluster which were added to make s-plex
                        for old_node in self.clusters[current_clust]:
                            edge = [node, old_node] if node < old_node else [old_node, node]
                            if edge in self.edges_modified:
                                delta -= self.weights[edge[0], edge[1]]
                                self.edges_modified.remove(edge)

                            # Add to solution all edges which were in the original graph between node and old elements
                            elif old_node in self.initial_neighbors[node]:
                                edge = [node, old_node] if node < old_node else [old_node, node]
                                delta += self.weights[edge[0], edge[1]]
                                self.edges_modified.append(edge)
                        # print(f"Preliminary manipulation 1 took {time.time() - time4}")


                        #time5 = time.time()
                        # Remove from solution all edges between node and elements in new cluster which were added to isolate it
                        for new_node in self.clusters[dest_clust]:
                            edge = [node, new_node] if node < new_node else [new_node, node]
                            if edge in self.edges_modified:
                                delta -= self.weights[edge[0], edge[1]]
                                self.edges_modified.remove(edge)
                        #print(f"Preliminary manipulation 2 took {time.time() - time5}")
                        
                        #time7 = time.time()
                        # Append new node
                        self.clusters[dest_clust].append(node)
                        current_clust = dest_clust
                        # self.update_current_neighbours()
                        # print(f"Now we have clusters:\n\t{self.clusters}")
                        # print(f"We start with these edges:\n\t{self.edges_modified}")

                        # Now we have to check and ensure the s-plex assumption is satisfied after adding new node.
                        # So we have to add edges again but without rebuilding the previous ones
                        
                        # Now we rebuild the s-plex for that cluster by only adding edges within

                        n_nodes = len(self.clusters[dest_clust])
                        cluster_neighbours = {x:[] for x in self.clusters[dest_clust]}

                        # Build a dictionary with node and its cluster neighbours
                        # print(f"NEW CLUSTER: {self.clusters[dest_clust]}")
                        # print(f"Initial neighbours: {self.initial_neighbors}")
                        for clust_node in self.clusters[dest_clust]:
                            cluster_neighbours[clust_node] = [x for x in self.initial_neighbors[clust_node] if x in self.clusters[dest_clust]]
                        #print(f"Cluster {self.clusters[dest_clust]}:\nNeighbours list {cluster_neighbours}")
                        
                        # Count number of neighbours for each
                        count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
                        nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                        #print(f"Number of neighbours for each node is {count_neighbours}")
                        #print(f"Neighbours not satisfying assumption are {nodes_not_satisfied}")
                        
                        # Consider list of potential edges to add
                        if len(nodes_not_satisfied) != 0:
                            potential_edges = []
                            # This is quite inefficient as more checks than necessary
                            for ind, node_i in enumerate(self.clusters[dest_clust]):
                                for node_j in self.clusters[dest_clust][ind+1 : ]:
                                    if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied: # only consider edges between unsatisfied nodes.
                                        if self.weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                                            edge_to_append = [node_i, node_j] if node_i < node_j else [node_j, node_i]
                                            potential_edges.append([edge_to_append ,self.weights[node_i, node_j]]) # [[node_i, node_j], weight]
                            potential_edges.sort(key=lambda x:x[1]) # Sort in decreasing order
                            # print(f"Potential edges: {potential_edges}")

                            while nodes_not_satisfied:
                                candidate_edge = potential_edges.pop(0)
                                node_i = candidate_edge[0][0]
                                node_j = candidate_edge[0][1]
                                cluster_neighbours[node_i].append(node_j)
                                cluster_neighbours[node_j].append(node_i)
                                count_neighbours[node_i] += 1
                                count_neighbours[node_j] += 1
                                nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                                #print(f"Adding edge between ({node_i}, {node_j})")
                                self.edges_modified.append(candidate_edge[0])  # Append additional edges we inserted
                                delta += self.weights[node_i, node_j]
                                # Have to update potential edges to only consider the nodes which are yet not satisfied
                                potential_edges = [item for item in potential_edges if item[0][0] in nodes_not_satisfied or item[0][1] in nodes_not_satisfied]

                        #print(f"Building the s-plex took {time.time() - time7}")
                        #print(f"Total time is {time.time() - time2}")
                        #return
                        #print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")
                        #print(f"Computed with delta evaluation our solution has value: {delta_baseline + delta}")
                        # self.update_current_neighbours()

                        if delta + delta_baseline < best_sol_value:
                            # Means we found a better solution
                            better_found = True
                            best_sol = self.copy()
                            best_sol_value = delta + delta_baseline
                            if step_function == 'first':
                                return better_found
     
                        delta_baseline = delta_baseline + delta

            self.copy_from(best_sol)
            return better_found
        elif step_function == 'random':
            delta = 0
            
            # Pick a node at random and move it to another cluster at random and see if it improved
            node = random.randint(1, len([x for clust in self.clusters for x in clust]))
            dest_clust = random.randint(0, len(self.clusters)-1)
            initial_cluster = [index for index, sublist in enumerate(self.clusters) if node in sublist][0]
            current_clust = initial_cluster
            while dest_clust == initial_cluster: # Generate until you obtain a new cluster diff than initial one
                dest_clust = random.randint(0, len(self.clusters)-1)
            
            self.clusters[current_clust].remove(node)
            # Remove from solution all edges between node and elements in old cluster which were added to make s-plex
            for old_node in self.clusters[current_clust]:
                edge = [node, old_node] if node < old_node else [old_node, node]
                if edge in self.edges_modified:
                    self.edges_modified.remove(edge)
                    delta = delta - self.weights[edge[0], edge[1]]

                # Add to solution all edges which were in the original graph between node and old elements
                elif old_node in self.initial_neighbors[node]:
                    edge = [node, old_node] if node < old_node else [old_node, node]
                    self.edges_modified.append(edge)
                    delta = delta + self.weights[edge[0], edge[1]]

            # Remove from solution all edges between node and elements in new cluster which were added to isolate it
            for new_node in self.clusters[dest_clust]:
                edge = [node, new_node] if node < new_node else [new_node, node]
                if edge in self.edges_modified:
                    self.edges_modified.remove(edge)
                    delta = delta - self.weights[edge[0], edge[1]]
            
            # Remove from solution all edges between nodes in the new cluster which were added to make s-plex as we have to rebuild it
            for ind1, new_node1 in enumerate(self.clusters[dest_clust]):
                for ind2, new_node2 in enumerate(self.clusters[dest_clust]):
                    edge = [new_node1, new_node2] if new_node1 < new_node2 else [new_node2, new_node1]
                    if edge in self.edges_modified:
                        self.edges_modified.remove(edge)
                        delta = delta - self.weights[edge[0], edge[1]]

            # Append new node
            self.clusters[dest_clust].append(node)
            current_clust = dest_clust
            # self.update_current_neighbours()
            # print(f"RANDOM: Now we have clusters:\n\t{self.clusters}")
            # print(f"We start with these edges:\n\t{self.edges_modified}")
            
            # Now we rebuild the s-plex for that cluster by only adding edges within
            n_nodes = len(self.clusters[dest_clust])
            cluster_neighbours = {x:[] for x in self.clusters[dest_clust]}

            # Build a dictionary with node and its cluster neighbours
            #print(f"NEW CLUSTER: {self.clusters[dest_clust]}")
            #print(f"Initial neighbours: {self.initial_neighbors}")
            for clust_node in self.clusters[dest_clust]: 
                cluster_neighbours[clust_node] = [x for x in self.initial_neighbors[clust_node] if x in self.clusters[dest_clust]]
            #print(f"Cluster {self.clusters[dest_clust]}:\nNeighbours list {cluster_neighbours}")
            
            # Count number of neighbours for each
            count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
            nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
            #print(f"Number of neighbours for each node is {count_neighbours}")
            #print(f"Neighbours not satisfying assumption are {nodes_not_satisfied}")
            
            # Consider list of potential edges to add
            if len(nodes_not_satisfied) != 0:
                potential_edges = []
                # This is quite inefficient as more checks than necessary
                for ind, node_i in enumerate(self.clusters[dest_clust]):
                    for node_j in self.clusters[dest_clust][ind+1 : ]:
                        if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied: # only consider edges between unsatisfied nodes.
                            if self.weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                                edge_to_append = [node_i, node_j] if node_i < node_j else [node_j, node_i]
                                potential_edges.append([edge_to_append ,self.weights[node_i, node_j]]) # [[node_i, node_j], weight]
                potential_edges.sort(key=lambda x:x[1]) # Sort in decreasing order
                

                while nodes_not_satisfied:
                    candidate_edge = potential_edges.pop(0)
                    node_i = candidate_edge[0][0]
                    node_j = candidate_edge[0][1]
                    cluster_neighbours[node_i].append(node_j)
                    cluster_neighbours[node_j].append(node_i)
                    count_neighbours[node_i] += 1
                    count_neighbours[node_j] += 1
                    nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - self.s]
                    #print(f"Adding edge between ({node_i}, {node_j})")
                    self.edges_modified.append(candidate_edge[0])  # Append additional edges we inserted
                    delta = delta + self.weights[candidate_edge[0][0], candidate_edge[0][1]]
                    # Have to update potential edges to only consider the nodes which are yet not satisfied
                    potential_edges = [item for item in potential_edges if item[0][0] in nodes_not_satisfied or item[0][1] in nodes_not_satisfied]


            if delta < 0:
                best_sol = self.copy()
                return True
            else:
                self.copy_from(best_sol)
                return False

    def local_search_move1node(self, par = None, result = Result()) -> None:
        start_time = time.time()
        print(f"Start Move1: current score: {self.calc_objective()}, step: {par}")
        result.changed = self.ls_move1node_simplified(par)
        print(f"End Move1: current score: {self.calc_objective()}, changed: {result.changed}, time: {time.time()-start_time}")

    def local_search_swap2nodes(self, par = None, result = Result()) -> None:
        start_time = time.time()
        print(f"Start Swap2: current score: {self.calc_objective()}, step: {par}")
        result.changed = self.ls_swap2nodes(par)
        print(f"End Swap2: current score: {self.calc_objective()}, changed: {result.changed}, time: {time.time()-start_time}")

    def local_search_join_clusters(self, par = None, result = Result()) -> None:
        start_time = time.time()
        print(f"Start Join: current score: {self.calc_objective()}, step: {par}")
        result.changed = self.ls_join_clusters(par)
        print(f"End Join: current score: {self.calc_objective()}, changed: {result.changed}, time: {time.time()-start_time}")

    def ch_construct_randomized(self, par, result):
        self.construct_randomized(k=par["k"], alpha=par["alpha"], beta=par["beta"])    
    
    def ch_construct(self, par, result):
        self.construct_randomized(k=par["k"], alpha=1, beta=1)

    def ls_move1node(self, step_function = "best") -> bool:
        """
        LOCAL SEARCH MOVE ONE NODE
        Performs one iteration of local search using the moving of one node from one cluster to another
        Returns True if an improevd solution is found
        """
        # Store the initial solution in here
        best_sol = self.copy()
        better_found = False

        if step_function in ['best', 'first']:
            # time1 = time.time()
            initial_clusters = dcopy(self.clusters)
            # print(f"Initialization took {time.time() - time1}s")
            
            for node in range(1, len(self.weights)):
                # time2 = time.time()
                self.clusters = dcopy(initial_clusters)
                # print(f"Working with node: {node}")
                initial_cluster = [index for index, sublist in enumerate(self.clusters) if node in sublist][0] # Find in what cluster it is
                current_cluster = initial_cluster
                # print(f"First deep copies took {time.time() - time2}s")
                # return
                # So now we move node to dest_clust in every iteration
                for dest_clust in range(len(self.clusters)):
                    if dest_clust != initial_cluster: 
                        # time3 = time.time()
                        # print(f"Moving node {node} from {self.clusters[current_cluster]} to {self.clusters[dest_clust]}")
                        self.clusters[current_cluster].remove(node)
                        self.clusters[dest_clust].append(node)
                        current_cluster = dest_clust
                        #print(f"Now we have clusters:\n\t{self.clusters}")
                        # Need to complete the s-plexes with these clusters and evaluate them
                        # This amounts to having to recompute the s-plex for the cluster we moved to
                        #   as the one we moved from is guaranteed to still be an s-plex 
                        
                        # Reset and recompute 
                        # (NOT NECESSARY WILL HAVE TO IMPROVE THIS LATER USING DELTA EVAL)
                        # You only have to update the new ones -> WORK ON THIS
                        self.edges_modified = []
                        # self.update_current_neighbours()
                        #print(f"The neighbours are now:\n\t{self.current_neighbours}") 
                        #print(f"Which should be equal to the initial ones:\n\t{self.initial_neighbors}")
                        # time5 = time.time()
                        self.construct_all_splex()
                        #print(f"Constructing the s-plex took {time.time() - time5}")
                        # self.update_current_neighbours()
                        # print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")
                        #print(f"Building whole solution took {time.time() - time3}")
                        # return 
                        # print(f"The best solution has value {best_sol.calc_objective()}")
                        if self.calc_objective() < best_sol.calc_objective():
                            # Means we found a better solution
                            better_found = True
                            best_sol = self.copy()
                            if step_function == 'first':
                                return better_found

            self.copy_from(best_sol)
            return better_found # Output if we are using 'best' as step function

        elif step_function == 'random':
            # Pick a node at random and move it to another cluster at random and see if it improved
            initial_clusters = dcopy(self.clusters)
            node = random.randint(1, len([x for clust in initial_clusters for x in clust]))
            dest_clust = random.randint(0, len(self.clusters)-1)
            initial_cluster = [index for index, sublist in enumerate(self.clusters) if node in sublist][0]
            while dest_clust == initial_cluster: # Generate until you obtain a new cluster diff than initial one
                dest_clust = random.randint(0, len(self.clusters)-1)
            #print(f"Moving node {node} to cluster {self.clusters[dest_clust]}")
            # Move the node to the new cluster and recompute the s-plexes.
            self.clusters[initial_cluster].remove(node)
            self.clusters[dest_clust].append(node)
            #print(f"Now we have clusters:\n\t{self.clusters}")

            self.edges_modified = []
            # self.update_current_neighbours()
            #print(f"The neighbours are now:\n\t{self.current_neighbours}") 
            #print(f"Which should be equal to the initial ones:\n\t{self.initial_neighbors}")
            self.construct_all_splex()
            # self.update_current_neighbours()
            #print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")

            if self.calc_objective() < best_sol.calc_objective():
                best_sol = self.copy()
                return True
            else:
                self.copy_from(best_sol)
                return False

    def ls_swap2nodes(self, step_function = 'best') -> bool:
        "Performs local search where the neighbour of one solution is given as the solution but with two nodes swapped"
        best_sol = self.copy()
        better_found = False
        initial_clusters = dcopy(self.clusters)

        # Consider all pairs of nodes and swap them
        if step_function in ['best', 'first']:
            for node1 in range(1, len(self.weights)):
                for node2 in range(node1 + 1, len(self.weights)):
                    # self.clusters = dcopy(initial_clusters) moved at the end so only called when we modify it.
                    clust_1 = [index for index, sublist in enumerate(self.clusters) if node1 in sublist][0] # What cluster the first node is in.
                    clust_2 = [index for index, sublist in enumerate(self.clusters) if node2 in sublist][0] # What cluster the second node is in.
                    if clust_1 != clust_2:
                        self.clusters[clust_1].remove(node1)
                        self.clusters[clust_2].remove(node2)
                        self.clusters[clust_1].append(node2)
                        self.clusters[clust_2].append(node1)

                        # Reset and recompute
                        self.edges_modified = []
                        # self.update_current_neighbours()

                        self.construct_all_splex()
                        # self.update_current_neighbours()

                        if self.calc_objective() < best_sol.calc_objective():
                            # Means we found a better solution
                            better_found = True
                            best_sol = self.copy()
                            if step_function == 'first':
                                return better_found
                        self.clusters = dcopy(initial_clusters)

            self.copy_from(best_sol)
            return better_found
        elif step_function == 'random':
            # Pick a node at random and move it to another cluster at random and see if it improved
            node1 = random.randint(1, len(self.weights))
            node2 = random.randint(1, len(self.weights))

            clust1 = [index for index, sublist in enumerate(self.clusters) if node in sublist][0] # What cluster the first node is in.
            clust2 = [index for index, sublist in enumerate(self.clusters) if node in sublist][0] # What cluster the first node is in.

            while clust2 == clust1:
                node2 = random.randint(1, len(self.weights))
                clust2 = [index for index, sublist in enumerate(self.clusters) if node in sublist][0] # What cluster the first node is in.

            # Swap the nodes and recompute the clusters
            
            self.clusters[clust_1].remove(node1)
            self.clusters[clust_2].remove(node2)
            self.clusters[clust_1].append(node2)
            self.clusters[clust_2].append(node1)
            #print(f"Now we have clusters:\n\t{self.clusters}")

            self.edges_modified = []
            # self.update_current_neighbours()
            #print(f"The neighbours are now:\n\t{self.current_neighbours}") 
            #print(f"Which should be equal to the initial ones:\n\t{self.initial_neighbors}")
            self.construct_all_splex()
            # self.update_current_neighbours()
            #print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")
            if self.calc_objective() < best_sol.calc_objective():
                best_sol = self.copy()
                return True
            else:
                self.copy_from(best_sol)
                return False
    
    def ls_join_clusters(self, step_function = 'best') -> bool:
        """
        Performs local search where the neighbour of one solution is given as the solution but with two clusters being joined together
        """
        best_sol = self.copy()
        better_found = False

        initial_clusters = dcopy(self.clusters)
        
        if step_function in ['best', 'first']:
            # Consider all possible pairs of clusters to be joined together
            for ind1, clust1 in enumerate(initial_clusters):
                for ind2, clust2 in enumerate(initial_clusters[ind1+1:]):
                    self.clusters = dcopy(initial_clusters) # For simplicity
                    self.clusters.remove(clust1)
                    self.clusters.remove(clust2)
                    self.clusters.append(clust1 + clust2)
                    #print(f"Now we have clusters: {self.clusters}")                
                    self.edges_modified = []
                    # self.update_current_neighbours()
                    self.construct_all_splex()
                    # self.update_current_neighbours()
                    #print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")

                    #print(f"The best solution has value {best_sol.calc_objective()}")
                    #print(f"Our solution has values {self.calc_objective()}")   
                    if self.calc_objective() < best_sol.calc_objective():
                        # Means we found a better solution
                        better_found = True
                        best_sol = self.copy()
                        if step_function == 'first':
                            return better_found
            self.copy_from(best_sol)
            return better_found
        elif step_function == 'random':
            # Select two clusters at random and join them 
            ind1 = random.randint(0, len(self.clusters)-1)
            ind2 = random.randint(0, len(self.clusters)-1)
            while ind1 == ind2:
                ind2 = random.randint(0, len(self.clusters)-1)
            clust1 = self.clusters[ind1]
            clust2 = self.clusters[ind2]

            self.clusters.remove(clust1)
            self.clusters.remove(clust2)
            self.clusters.append(clust1 + clust2)
            #print(f"Now we have clusters: {self.clusters}")

            self.edges_modified = []
            # self.update_current_neighbours()
            self.construct_all_splex()
            # self.update_current_neighbours()
            #print(f"The edges modified are now:\n\t{self.edges_modified}\nAnd our solution has value:\n\t{self.calc_objective()}")
            if self.calc_objective() < best_sol.calc_objective():
                best_sol = self.copy()
                return True
            else:
                self.copy_from(best_sol)
                return False

if __name__ == '__main__':
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi_sol = SPlexSolution(spi)
    spi_sol.construct_randomized(k=100, alpha=1, beta=1)
    #for clust in spi_sol.clusters:
    #    print(f"Cluster has lenght {len(clust)}")
    print(f"Greedy solution has value: {spi_sol.calc_objective()} and is valid? {spi_sol.check()}")

    time_start = time.time()
    print(f"move1node_simplified: Did we find a better solution: {spi_sol.ls_join_clusters(step_function='best')}")
    print(f"It has value {spi_sol.calc_objective()} and is valid? {spi_sol.check()}")
    print(f"It took {time.time() - time_start}")

   
    

