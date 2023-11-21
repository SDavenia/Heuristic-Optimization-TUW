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

from construction import SPlexInstance

parser = get_settings_parser()



class VNDInstance:
    def __init__(self, filename, k):
        self.constructionInstance = SPlexInstance(filename=filename)
        self.constructionInstance.construct()
        self.x = self.constructionInstance.getClusters()
        
    def n1(self):
        exit()

    def start(self):
        print(self.x)

if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    parser = get_settings_parser()
    parser.add_argument("inputfile")
    args = parser.parse_args()
    vndi = VNDInstance(filename=args.inputfile, k=3)
    vndi.start()
