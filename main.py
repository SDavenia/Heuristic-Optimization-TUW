"""A general variable neighborhood search class which can also be used for plain local search, VND, GRASP, IG etc.

It extends the more general scheduler module/class by distinguishing between construction heuristics, local
improvement methods and shaking methods.
"""

from typing import List
import time

from pymhlib.scheduler import Method, Scheduler
from pymhlib.settings import get_settings_parser, settings, parse_settings
from pymhlib.solution import Solution
from pymhlib.gvns import GVNS
from SplexSolution import SPlexSolution
from SPlexInstance import SPlexInstance


parser = get_settings_parser()

def custom_settings():
    parser.add_argument("inputfile", type=str, help='instance file path')
    parser.add_argument("--alg", type=str, default="c", help='(heuristic) c: construction, rc: random construction, ls: local search, grasp, vnd, gvns)')
    parser.add_argument("--nh", type=str, default="m1", help='(local search neighborhood) m1: move 1 node, s2: swap 2 nodes, sc: split clusters, jc: join clusters')
    parser.add_argument("--ls_step", type=str, default="best", help='(local search step function) f: first, b: best, r: random')

def construction(solution: SPlexSolution):
    k = 100
    solution.construct_deterministic(k=k, cluster_size_cap=(sol.inst.n/k)*25)
    pass

def rand_construction(solution: SPlexSolution):
    k = 100
    solution.construct_randomized(k=k, alpha=0.5, beta=0.5, cluster_size_cap=(sol.inst.n/k)*25)
    pass

def local_seach(solution: SPlexSolution):
    k = 100
    vnd = GVNS(solution,[Method("random_construction", SPlexSolution.ch_construct_randomized,k)], 
                        [Method("local_search_move1node", SPlexSolution.local_search_move1node,"first")], 
                        [])
    vnd.run()

def grasp(solution: SPlexSolution):
    pass

def vnd(solution: SPlexSolution):
    k = 100
    vnd = GVNS(solution,[Method("random_construction", SPlexSolution.ch_construct_randomized,k)], 
                        [Method("local_search_move1node", SPlexSolution.local_search_move1node,"first")], 
                        [])
    vnd.run()

def gvns(solution: SPlexSolution):
    pass

def run(args, solution: SPlexSolution):
    if args.alg == "c":
        construction(solution)
    elif args.alg == "rc":
        rand_construction(solution)
    elif args.alg == "ls":
        local_seach(solution)
    elif args.alg == "grasp":
        grasp(solution)
    elif args.alg == "vnd":
        vnd(solution)
    elif args.alg == "gvns":
        gvns(solution)

if __name__ == '__main__':
    custom_settings()    
    parse_settings()
    spi = SPlexInstance(settings.inputfile)
    sol = SPlexSolution(spi)
    start_time = time.time()
    run(settings, sol)
    runtime = time.time() - start_time
    score = sol.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {sol.check()}")
