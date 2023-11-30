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

start_time = 0

def custom_settings():
    parser.add_argument("inputfile", type=str, help='instance file path')
    parser.add_argument("--alg", type=str, default="c", help='(heuristic) c: construction, rc: random construction, ls: local search, grasp, vnd, gvns)')
    parser.add_argument("--nh", type=str, default="m1", help='(local search neighborhood) m1: move 1 node, s2: swap 2 nodes, sc: split clusters, jc: join clusters')
    parser.add_argument("--ls_step", type=str, default="best", help='(local search step function) f: first, b: best, r: random')
    parser.add_argument("--k", type=int, default=5, help='construction initial cluster size')
    parser.add_argument("--alpha", type=float, default=0.5, help='randomization for cluster initialization')
    parser.add_argument("--beta", type=float, default=0.5, help='randomization for cluster assignment')
    parser.add_argument("--step", type=str, default="first", help='step function selection')
    parser.add_argument("--iterations", type=int, default=5, help='iterations, for e.g. grasp')

def construction(solution: SPlexSolution, k):
    start_time = time.time()
    solution.construct_deterministic(k=k)
    runtime = time.time() - start_time
    score = solution.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {solution.check()}")

def rand_construction(solution: SPlexSolution, k, alpha, beta):
    start_time = time.time()
    solution.construct_randomized(k=k, alpha=alpha, beta=beta)
    runtime = time.time() - start_time
    score = solution.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {solution.check()}")

def local_seach(solution: SPlexSolution, k, alpha, beta, step):
    start_time = time.time()
    par = {"k": k, "alpha": alpha, "beta": beta}
    ls = GVNS(solution,[Method("random_construction", SPlexSolution.ch_construct_randomized, par)], 
                        [Method("local_search_move1node", SPlexSolution.local_search_move1node, step)], 
                        [])
    ls.run()
    runtime = time.time() - start_time
    score = solution.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {solution.check()}")

def grasp(solution: SPlexSolution, k, alpha, beta, step, iterations):
    start_time = time.time()
    par = {"k": k, "alpha": alpha, "beta": beta}
    best_score = float("inf")
    best_solution = None
    for i in range(iterations):
        print(f"GRASP iteration {i+1} of {iterations}")
        new_solution = solution.copy()
        ls = GVNS(new_solution,[Method("random_construction", SPlexSolution.ch_construct_randomized, par)], 
                        [Method("local_search_swap2nodes", SPlexSolution.local_search_swap2nodes, step)], 
                        [])
        ls.run()
        new_score = new_solution.calc_objective()
        print(f"GRASP iteration finished. score: {new_score}, best: {best_score}")
        if new_score < best_score:
            best_score = new_score
            best_solution = new_solution
    runtime = time.time() - start_time
    score = best_solution.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {best_solution.check()}")

def vnd(solution: SPlexSolution, k, alpha, beta, step):
    start_time = time.time()
    par = {"k": k, "alpha": alpha, "beta": beta}
    vnd = GVNS(solution,[Method("random_construction", SPlexSolution.ch_construct_randomized, par)], 
                        [Method("local_search_join_clusters", SPlexSolution.local_search_join_clusters, step), 
                         Method("local_search_swap2nodes", SPlexSolution.local_search_swap2nodes, step), 
                         Method("local_search_move1node", SPlexSolution.local_search_move1node, step)], 
                        [])
    vnd.run()
    runtime = time.time() - start_time
    score = solution.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {solution.check()}")

def gvns(solution: SPlexSolution, k, alpha, beta, step):
    start_time = time.time()
    par = {"k": k, "alpha": alpha, "beta": beta}
    gvns = GVNS(solution,[Method("random_construction", SPlexSolution.ch_construct_randomized, par)], 
                        [Method("local_search_join_clusters", SPlexSolution.local_search_join_clusters, step), 
                         Method("local_search_swap2nodes", SPlexSolution.local_search_swap2nodes, step), 
                         Method("local_search_move1node", SPlexSolution.local_search_move1node, step)], 
                        [Method("shake_join_clusters", SPlexSolution.shake_join_clusters, None )])
    gvns.run()
    runtime = time.time() - start_time
    score = solution.calc_objective()
    print(f"Runtime: {runtime}\nScore: {score}\nSolution check: {solution.check()}")

def run(args, solution: SPlexSolution):
    if args.alg == "c":
        construction(solution, args.k)
    elif args.alg == "rc":
        rand_construction(solution, args.k, args.alpha, args.beta)
    elif args.alg == "ls":
        local_seach(solution, args.k, args.alpha, args.beta, args.step)
    elif args.alg == "grasp":
        grasp(solution, args.k, args.alpha, args.beta, args.step, args.iterations)
    elif args.alg == "vnd":
        vnd(solution, args.k, args.alpha, args.beta, args.step)
    elif args.alg == "gvns":
        gvns(solution, args.k, args.alpha, args.beta, args.step)

if __name__ == '__main__':
    custom_settings()    
    parse_settings()
    spi = SPlexInstance(settings.inputfile)
    sol = SPlexSolution(spi)
    run(settings, sol)
    
