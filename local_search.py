import random
from typing import Any, Tuple
from itertools import combinations
import heapq
import networkx as nx
import copy


from pymhlib.solution import SetSolution, TObj
from pymhlib.settings import get_settings_parser
from pymhlib.scheduler import Result
from pymhlib.demos.graphs import create_or_read_simple_graph

import numpy
import bisect

from SPlexInstance import SPlexInstance

parser = get_settings_parser()

"""
Takes as input the set of clusters
"""

class LocalSearchInstance:
    def __init__(self, clusters):
        self.initial_clusters = clusters
        

    @staticmethod 
    def move1_nhour(sol, _par, result):
        """
        Define it as a generator that gives one element from the neighbourhood at a time
        """
        # Consider moving each node to each possible cluster
        n_nodes = sum([len(clust) for clust in _par])
        n_clusters = len(_par)

        # Consider moving each node to each possible cluster
        for ind, clust in enumerate(_par):
            for node in clust:
                for ind2, clust2 in enumerate(_par):
                    if ind != ind2:
                        proposed_solution = copy.deepcopy(_par)
                        proposed_solution[ind].remove(node)  # Remove node from its cluster
                        proposed_solution[ind2].append(node) # Add node to proposed cluster
                        yield(proposed_solution)



if __name__ == '__main__':
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    parser.add_argument("step_function")
    
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi.construct(k=2)

    # Extract the list of clusters
    initial_clusters = spi.clusters











"""
@staticmethod
    def evaluate_solution(problem_file, solution_file):
        # Extract (i, j) pairs contained in the solution file
        solution_pairs = []     
        with open(solution_file) as file:
            lines = file.readlines()
            for line in lines[1:]:
                values = line.split()
                solution_pairs.append(values[0] + ' ' + values[1])
        
        cost = 0
        counter = 0 # Variable to check if we found all edges in the solution, if not there are some problems.
        
        # Extract cost by looking at the weights in the problem file    
        with open(problem_file, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                values = line.split()
                entry = values[0] + ' ' + values[1]
                if entry in solution_pairs:
                    cost += int(values[3])
                    counter += 1 
        
        # Additional check to ensure that all the edges in the solution have an associated weight in the problem instance
        if counter != len(solution_pairs):
            print(f"Something wrong, {len(solution_pairs) - counter} edges in the solution could not be found")
            return 
        return(cost)


"""