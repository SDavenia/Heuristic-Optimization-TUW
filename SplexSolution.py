from pymhlib.solution import Solution, TObj
from pymhlib.settings import get_settings_parser


from SPlexInstance import SPlexInstance

from operator import itemgetter
import random
import numpy

from copy import deepcopy as dcopy

class SPlexSolution(Solution):
    """
    A solution of the s-plex editing problem.
    It contains the following attributes
        - s: plex order desired
        - weights: instance weights
        - weights_given_graph: weights of edges already present.
        - initial_neighbors: initial neighbourhood list for the nodes
        - current_neighbors: neighborhood list given the current solution
        - clusters: set of clusters of the solution.
        - edges_modified: solution representation, contains what edges were removed/added from the initial list
    """
    to_maximise = False
    
    def __init__(self, inst: SPlexInstance):
        super().__init__()
        self.inst = inst
        self.s = inst.s
        self.weights = inst.weights
        self.weights_given_graph = inst.weights_given_graph
        self.initial_neighbors = inst.neighbors_given_graph
        self.current_neighbours = self.initial_neighbors

        # edges_modified contains the solution which will be written to file.
        # clusters is used to help as it is often used for the neighbourhood structures as well.
        self.clusters = []
        self.edges_modified = []

    def calc_objective(self)->int:
        cost = 0
        for edge in self.edges_modified:
            cost += self.weights[edge[0], edge[1]]
        print(f"The given solution has cost: {cost}")
        return cost
    
    def __repr__(self):
        raise NotImplementedError
    
    def update_current_neighbours(self):
        """
        It updates the values of current_neighbours given the current solution.
        """
        self.current_neighbours = self.initial_neighbors.copy()
        #print(f"We applied these modifications in our solution {self.edges_modified}")
        for edge in self.edges_modified:
            if edge[1] in self.initial_neighbors[edge[0]]:  #The edge was present initially, so we removed it
                self.current_neighbours[edge[1]].remove(edge[0])
                self.current_neighbours[edge[0]].remove(edge[1])
            else:                                           #The edge was NOT present initially, so we added it
                self.current_neighbours[edge[1]].append(edge[0])
                self.current_neighbours[edge[0]].append(edge[1])
        #print(f"Our solution contains these neighbours\n\t{self.current_neighbours}")


    def check(self):
        # We check if all our clusters are s-plexes and if there are no edges between them
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
        print(f"VALID SOLUTION")
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
    
    def construct_set_initial(self, k, alpha):
        """
        Selects the initial k nodes for the clusters with a level of randomization alpha
        Returns sorted_nodes which is required for the construction
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
            neighbors_selected_nodes += self.initial_neighbors[selected_node] # Add neighours of that node to it

        print(f"Selected nodes: {selected_nodes}")
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
        while node_similarity_to_cluster:
            node_max = {max(value) for value in node_similarity_to_cluster.values()} # Highest similarity for each node
            node_min = {min(value) for value in node_similarity_to_cluster.values()}
            cmax = max(node_max) 
            cmin = min(node_min)
            #print(f"Cmax: {cmax}, Cmin: {cmin}")
            threshold = cmin + beta*(cmax - cmin)

            # Consider all pairings of node-cluster which are above the threshold
            RCL_pairs = [(key, element) for key, values in node_similarity_to_cluster.items() for element in values if element >= threshold]
            #print(f"Result threshold {threshold} pairs: {RCL_pairs}")
            if beta == 1:
                node_assigned = max(node_similarity_to_cluster, key=lambda k: max(node_similarity_to_cluster[k]))  # Node with highest similarity chosen deterministically (first in list)
                cluster_assigned, _ = max(enumerate(node_similarity_to_cluster[node_assigned]), key=itemgetter(1))
            else:
                pair_assigned = RCL_pairs.pop(random.randint(0, len(RCL_pairs)-1))
                node_assigned = pair_assigned[0]
                #print(f"NODE {node_similarity_to_cluster[node_assigned]}")
                cluster_assigned = [index for index, value in enumerate(node_similarity_to_cluster[node_assigned]) if value == pair_assigned[1]][0] #[0] To remove duplicates if present


            #print(f"Assigning node {node_assigned} to cluster {cluster_assigned}")
            self.clusters[cluster_assigned].append(node_assigned)
            del node_similarity_to_cluster[node_assigned]
            for node in node_similarity_to_cluster.keys():
                if self.weights_given_graph[node, node_assigned] != 0:
                    node_similarity_to_cluster[node][cluster_assigned] += self.weights[node, node_assigned]
                else:
                    node_similarity_to_cluster[node][cluster_assigned] -= self.weights[node, node_assigned]
        print(f"Final clusters:\n\t{self.clusters}")

    def construct_splex(self):
        # We have to count the number of edges within the s-plex and add some where required
        # A solution is represented by the number of edges which have been changed from the original graph.

        solution_edges = []  # Use a list of sets to identify duplicates such as (i, j) and (j, i)
        for clust in self.clusters:

            # Include in solution the edges we removed to create the clusters:
            for node in clust:
                non_cluster_neighbours = [x for x in self.initial_neighbors[node] if x not in clust]
                if non_cluster_neighbours:
                    edges_removed = [set([node, x]) for x in non_cluster_neighbours]
                    #print(f"For node {node} we removed edges {edges_removed}")
                    solution_edges += edges_removed
                    

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
                    solution_edges.append(set([node_i, node_j]))  # Append additional edges we inserted
                    #print(f"Nodes which do not satisfy are {nodes_not_satisfied}")

        # Now we have to add to the solution the edges we removed from the original graph, i.e. the ones between clusters

        # Return a solution by removing duplicate
        self.edges_modified = list(set(map(frozenset, solution_edges)))
        self.edges_modified = [sorted(list(fs)) for fs in self.edges_modified]
        print(f"Solution edges: {self.edges_modified}")

    
    def construct_randomized(self, k, alpha, beta):
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
        self.construct_splex()
        self.update_current_neighbours() # Called to update current neighbours as well given the solution found so far



        

    
    def construct_deterministic(self, k):
        return self.construct_randomized(k, alpha=1, beta=1) # or something similar where alpha is a probabilistic parameter


            

if __name__ == '__main__':
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi_sol = SPlexSolution(spi)
    spi_sol.construct_randomized(k=3, alpha=1, beta=1)
    # spi_sol.construct_deterministic(k=3)
    spi_sol.check()
    spi_sol.calc_objective()