from pymhlib.settings import get_settings_parser

import re
from utilities import Utilities

parser = get_settings_parser()

class SPlexInstance:
    """s-Plex problem instance

    Given an undirected simple graph, find the minimal weight of edited 
    edges to retain a graph of unconnected s-Plexes

    Attributes
        - s: s-Plex size
        - n: number of nodes
        - m: number of edges
        - weights: nxn matrix filled with weights of all possible edges
        - neighbors: list of neighbours for each node for the given graph
    """

    def __init__(self, filename):
        """Create or read graph with given name."""
        graph = self.common = Utilities.importFile(filename=filename)
        self.s = graph["s"]
        self.n = graph["n"]
        self.m = graph["m"]
        self.weights = graph["weights"]
        self.weights_given_graph = graph["weights_given_graph"]     # Redundant, might remove it later
        self.neighbors_given_graph = graph["neighbors_given_graph"]
        
        # Extract problem instance which we will have to write in solution file.
        pattern = r'.*/(.*?)\..*$'
        self.problem_instance = re.sub(pattern, r'\1', filename)
    
