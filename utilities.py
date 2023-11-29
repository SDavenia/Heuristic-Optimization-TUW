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
            remove_costs = numpy.zeros((n + 1, n + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
            add_costs = numpy.zeros((n + 1, n + 1), numpy.int8) # add 1 because input file starts nodes with 1 (not 0)
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
                    remove_costs[a][b] = w
                    remove_costs[b][a] = w
                    neighbors_given_graph[a].append(b)
                    neighbors_given_graph[b].append(a)
                else:
                    add_costs[a][b] = w
                    add_costs[b][a] = w
        return({"s": s, "n": n, "m": m, "weights": weights, "weights_given_graph": weights_given_graph, 
                "neighbors_given_graph": neighbors_given_graph, "remove_costs": remove_costs, "add_costs": add_costs})


    @staticmethod
    def write_solution(filename, solution, header):
        """
        Takes as input a 
            - filename where to write solution (in a form like "instance_type/instance_name.txt")
            - unique_solution: to write to file
        Writes a file containing a header which should be the instance name and the number of edges in solution.
        This file is written to solutions/instance_type/instance_name.txt
        """
        filename = 'solutions/' + filename
        with open(filename, 'w') as file_:
            file_.write(header + "\n")
            for item in solution:
                item = str(item).replace("[", "").replace("]", "").replace(",", "")
                file_.write(item + '\n')

