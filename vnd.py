"""A general variable neighborhood search class which can also be used for plain local search, VND, GRASP, IG etc.

It extends the more general scheduler module/class by distinguishing between construction heuristics, local
improvement methods and shaking methods.
"""

from typing import List
import time

from pymhlib.scheduler import Method, Scheduler
from pymhlib.settings import get_settings_parser, settings, parse_settings
from pymhlib.solution import Solution
from SplexSolution import SPlexSolution
from SPlexInstance import SPlexInstance
from local_search import LocalSearchInstance


parser = get_settings_parser()

class VND(Scheduler):
    """A general variable descent (VND).

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meths_li: list of local improvement methods
    """

    to_maximise = False

    def __init__(self, sol: Solution, meths_li: List[Method], own_settings: dict = None, consider_initial_sol=True):
        """Initialization.

        :param sol: solution to be improved
        :param meths_li: list of local improvement methods
        """
        super().__init__(sol, meths_li, own_settings, consider_initial_sol)
        self.meths_li = meths_li

    def vnd(self, sol: Solution) -> bool:
        """Perform variable neighborhood descent (VND) on given solution.

        :returns: true if a global termination condition is fulfilled, else False.
        """
        sol2 = sol.copy()
        while True:
            for m in self.next_method(self.meths_li):
                res = self.perform_method(m, sol2)
                if sol2.is_better(sol):
                    sol.copy_from(sol2)
                    if res.terminate:
                        return True
                    break
                if res.terminate:
                    return True
                if res.changed:
                    sol2.copy_from(sol)
            else:  # local optimum reached
                return False

    def run(self) -> None:
        """Actually performs the VND."""
        sol = self.incumbent.copy()
        self.vnd(sol)

if __name__ == '__main__':
    parser = get_settings_parser()

    if not settings.__dict__: parse_settings(args='')

    parser.add_argument("inputfile")
    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    spi_sol = SPlexSolution(spi)
    spi_sol.construct_randomized(k=3, alpha=1, beta=1)
    vnd = VND(spi_sol, [Method("move1_nhour", LocalSearchInstance.move1_nhour, spi_sol.edges_modified)])
    vnd.run()