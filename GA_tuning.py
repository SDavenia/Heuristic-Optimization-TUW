import pyhopper
import os

from pymhlib.solution import Solution, TObj
from pymhlib.settings import get_settings_parser
from pymhlib.scheduler import Result
from SplexSolution import SPlexSolution
from SplexSolution import SPlexInstance

from copy import deepcopy as dp
import numpy as np
import random
import time

from GA import GeneticAlgorithm


def my_objective(params: dict) -> float:
    """
    Takes as input a problem instance and a set of parameters and returns the objective value.
    """
    spi = SPlexInstance(args.inputfile)
    k_use = int(spi.n * params['k_perc_n'] + spi.m/spi.n * params['k_perc_mn'])
    GA_instance = GeneticAlgorithm(spi, params['n_solutions'], k_use, params['alpha'], params['beta'], params['selection_method'], params['perc_replace'], params['join_p_param'])
    GA_instance.generate_initial_population()
    GA_instance.evolve_n_steps(n=20)
    return GA_instance.population_values[0]


if __name__ == '__main__':
    parser = get_settings_parser()
    parser.add_argument("inputfile", type=str, help='instance file path')
    parser.add_argument("--runtime", type=str, default = '1m', help='runtime for the tuning procedure')
    parser.add_argument("--n_jobs", type=int, default = 1, help='Amount of parallelisation for tuning')
    args = parser.parse_args()
    print(f"Runtime: {args.runtime}")

    search = pyhopper.Search(
        n_solutions = pyhopper.int(10, 20, multiple_of=10),
        alpha = pyhopper.float(0.1, 0.5, "0.1f"),
        beta = pyhopper.float(0.5, 1.0, "0.1f"),
        k_perc_n = pyhopper.choice([0.1, 0.15, 0.20]),
        k_perc_mn = pyhopper.float(0.1, 0.5, "0.1f"),
        selection_method = pyhopper.choice(["lr","fp"]),
        perc_replace = pyhopper.float(0.7, 0.9, "0.1f"),
        join_p_param = pyhopper.int(10, 40, multiple_of=10)
    )
    best_params = search.run(my_objective, "minimize", runtime=args.runtime, n_jobs=args.n_jobs, quiet=True)
    print(best_params)

    