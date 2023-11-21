import numpy

class Utilities:
    @staticmethod
    def importFile(filename):
        with open(filename, "r") as infile:
            graph_params = infile.readline().split()
            s = int(graph_params[0])
            n = int(graph_params[1])
            m = int(graph_params[2])
            weights = numpy.zeros((n + 1, n + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
            weights_given_graph = numpy.zeros((n + 1, n + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
            neighbors_given_graph = dict.fromkeys(range(1,n + 1))
            for k, _ in neighbors_given_graph.items(): 
                neighbors_given_graph[k] = []
            for line in infile.readlines():
                (a,b,e,w) = [int(x) for x in line.split()]
                weights[a][b] = w
                weights[b][a] = w
                if e == 1:
                    weights_given_graph[a][b] = w
                    weights_given_graph[b][a] = w
                    neighbors_given_graph[a].append(b)
                    neighbors_given_graph[b].append(a)
        return({"s": s, "n": n, "m": m, "weights": weights, "weights_given_graph": weights_given_graph, "neighbors_given_graph": neighbors_given_graph})


    ########## TURN THE CLUSTERS INTO S-PLEXES ##########
    # We have to count the number of edges within the s-plex and add some where required

    # A solution is represented by the number of edges which have been changed from the original graph.
    @staticmethod
    def buildSplexesFromClusters(clusters, weights, neighbors_given_graph, weights_given_graph):
        solution_edges = []  # Use a list of sets to identify duplicates such as (i, j) and (j, i)

        for clust in clusters:

            # Include in solution the edges we removed to create the clusters:
            for node in clust:
                non_cluster_neighbours = [x for x in neighbors_given_graph[node] if x not in clust]
                if non_cluster_neighbours:
                    edges_removed = [set([node, x]) for x in non_cluster_neighbours]
                    print(f"For node {node} we removed edges {edges_removed}")
                    solution_edges += edges_removed
                    

            n_nodes = len(clust)
            cluster_neighbours = {node:[] for node in clust}
            for node in clust:
                cluster_neighbours[node] = [x for x in neighbors_given_graph[node] if x in clust]
            print(f"Cluster {clust}:\nNeighbours list {cluster_neighbours}")

            # Now we need the order of each node in the subgraph
            count_neighbours = {key:len(value) for key,value in cluster_neighbours.items()}
            print(f"Number of neighbours for each node is {count_neighbours}")

            # List of nodes which do not have order for the s-plex assumption
            nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - s]
            print(f"Nodes which do not satisfy are {nodes_not_satisfied}")
            
            # Now we create a list of potential edges to add where we only consider pairs where at least one of the nodes is a unsatisfactory one
            # Now we add edges with minimum cost at every iteration
            if len(nodes_not_satisfied) != 0:
                potential_edges = []
                # This is quite inefficient as more checks than necessary
                for ind, node_i in enumerate(clust):
                    for node_j in clust[ind+1 : ]:
                        if node_i in nodes_not_satisfied or node_j in nodes_not_satisfied: # only consider edges between unsatisfied nodes.
                            if weights_given_graph[node_i, node_j] == 0: # means it is not in the given graph
                                potential_edges.append([[node_i, node_j],weights[node_i, node_j]]) # [[node_i, node_j], weight]
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
                    nodes_not_satisfied = [x for x in count_neighbours.keys() if count_neighbours[x] < n_nodes - s]
                    print(f"Adding edge between ({node_i}, {node_j})")
                    solution_edges.append(set([node_i, node_j]))  # Append additional edges we inserted
                    print(f"Nodes which do not satisfy are {nodes_not_satisfied}")

        # Now we have to add to the solution the edges we removed from the original graph, i.e. the ones between clusters

        # Return a solution by removing duplicate
        unique_solutions = list(set(map(frozenset, solution_edges)))
        unique_solutions = [set(fs) for fs in unique_solutions]
        print(f"Solution edges: {unique_solutions}")
        return unique_solutions

