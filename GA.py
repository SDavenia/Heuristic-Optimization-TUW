from pymhlib.solution import Solution, TObj
from pymhlib.settings import get_settings_parser
from pymhlib.scheduler import Result
from SplexSolution import SPlexSolution
from SplexSolution import SPlexInstance

from copy import deepcopy as dp
import numpy as np
import random
import time

class GeneticAlgorithm():
    """
    Implementation of a GA class for the S-plex editing problem.
    Attributes:
        - problem_instance: stores the original problem instance.
        - n_solutions: the size of the population to consider.
        - population/population_values: the population of solutions and their values.
        - children/children_values: the newly generated children and their values.
        - alpha, beta, k: parameters for the initial random generation of solutions.
        - perc_replace: what percentage of the original population should be replaced.
        - selection: if fp uses fitness proportional selection, if lr it uses linear ranking with alpha=2, beta=0
    """
    def __init__(self, problem_instance, n_solutions, k, alpha, beta, selection_method, perc_replace, avg_joins):
        self.problem_instance = problem_instance
        self.n_solutions = n_solutions

        self.population = [None] * self.n_solutions
        self.population_values = [None] * self.n_solutions

        self.children = [None] * self.n_solutions # Assuming we generate again the whole population
        self.children_values = [None] * self.n_solutions

        self.initial_n_clusters = [None] * self.n_solutions # Stores the initial number of clusters in the solutions

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.perc_replace = perc_replace

        self.selection_method=selection_method
        self.avg_joins = avg_joins

    def generate_initial_population(self):
        """
        Generate initial population with variable size
        """
        # Prepare a list containing the different solutions
        self.population = [SPlexSolution(self.problem_instance) for i in range(self.n_solutions)]

        # Generate the initial population using a fixed value of k (for now, later maybe randomize it) 
        [self.population[i].construct_randomized(k = self.k, alpha=self.alpha, beta=self.beta) for i in range(self.n_solutions)]
        
        # Evaluate the initial population and sort it accordingly.
        self.evaluate_population()
        self.sort_population()
        self.initial_n_clusters = [len(clust) for ind in self.n_solutions for clust in self.children[ind].clusters]

        # Store the best value and the best solution: not needed cause they always improve
        # self.best_value = self.population_values[0]
        # self.best_solution = self.population[0]

    
    def sort_population(self):
        """
        Given the current values of the population, it re-orders the solutions accordingly, so that in position 1 we have the best solution and so on.
        """
        sorted_data = sorted(zip(self.population_values, self.population))
        self.population = [x for _, x in sorted_data]
        self.population_values = [y for y, _ in sorted_data]
    
    def sort_children(self):
        """
        Given the current values of the population, it re-orders the solutions accordingly, so that in position 1 we have the best solution and so on.
        """
        sorted_data = sorted(zip(self.children_values, self.children))
        self.children = [x for _, x in sorted_data]
        self.children_values = [y for y, _ in sorted_data]
    
    def evaluate_population(self):
        """
        Evaluate the current population by storing its values in a vector
        """
        self.population_values = [self.population[i].calc_objective() for i in range(self.n_solutions)]

    def evaluate_children(self):
        """
        Evaluate the current children by storing its values in a vector
        """
        self.children_values = [self.children[i].calc_objective() for i in range(self.n_solutions)]

    def select(self):
        if self.selection_method == 'lr':
            return select_linear_ranking
        if self.selection_method == 'fp':
            return select_linear_ranking
        raise ValueError("Selection method should be one of [lr, fp]")

    def select_fitness_prop(self):
        """
        Returns the index of two elements from the population to use for recombination, which are selected according fitness roulette.
        """
        den = 1/sum(self.population_values)
        selection_probabilities = [self.population_values[i] * den for i in range(self.n_solutions)]
        selected = np.random.choice(np.arange(0, self.n_solutions), size = 2, p = selection_probabilities, replace=False)
        return selected[0], selected[1]
    
    def select_linear_ranking(self):
        """
        Returns the index of two elements from the population to use for recombination, which are selected according to linear ranking
        Assumes that the population is already sorted in decreasing order.
        """
        alpha = 2
        beta = 0
        selection_probabilities = [(alpha + i * (beta-alpha)/(self.n_solutions-1))/self.n_solutions for i in range(self.n_solutions)]
        selected = np.random.choice(np.arange(0, self.n_solutions), size = 2, p = selection_probabilities, replace=False)
        return selected[0], selected[1]
    
    def recombine(self):
        """
        Generates n_solutions starting from the initial population by applying recombination.
        It selects cluster assignments at random from either of the two parents.
        """
        # self.children = [SPlexSolution(self.problem_instance) for i in range(self.n_solutions)] # Initialize children population
        # Generate all new individuals
        for ind in range(self.n_solutions):
            # Select two parents
            p0, p1 = self.select_fitness_prop()
            clusters_p0 = dp(self.population[p0].clusters)
            clusters_p1 = dp(self.population[p1].clusters)   
            # print(f"Initial clusters for the two parents:\n{clusters_p0}\n\n{clusters_p1}")         
            # print(f"Parent 0 has {len(clusters_p0)} clusters\nParent 1 has {len(clusters_p1)} clusters")
            child = SPlexSolution(self.problem_instance)
            
            # Choose at random a cluster from one of the parents and append it to the child solutions.
            # Keep selecting clusters until we are finished.
            while clusters_p0 and clusters_p1:
                parent_ind = random.randint(0, 1)
                if parent_ind==0:
                    cluster_ind = random.randint(0, len(clusters_p0)-1)
                    selected = clusters_p0[cluster_ind]
                    child.clusters.append(selected)
                    # Remove duplicates in other parents clusters:
                    clusters_p1 = [[item for item in sublist if item not in selected] for sublist in clusters_p1 if [item for item in sublist if item not in selected]]
                    del clusters_p0[cluster_ind]
                else:
                    cluster_ind = random.randint(0, len(clusters_p1)-1)
                    selected = clusters_p1[cluster_ind]
                    child.clusters.append(selected)
                    # Remove duplicates in other parents clusters:
                    clusters_p0 = [[item for item in sublist if item not in selected] for sublist in clusters_p0 if [item for item in sublist if item not in selected]]
                    del clusters_p1[cluster_ind]
                # print(f"Now parent clusters are:\n{clusters_p0}\n\n{clusters_p1}")
            # print(f"Now child clusters are: {child.clusters}")
            self.children[ind] = child
            # Check that all nodes are assigned to some cluster
            # print(f"Number of clusters: {len(self.children[ind].clusters)}")
            # print(len(set([x for sublist in self.children[ind].clusters for x in sublist])))
    
    def mutate(self):
        for ind in range(self.n_solutions):
            cluster_lengths = [len(clust) for clust in self.children[ind].clusters]
            n_clust = len(cluster_lengths)
            # If the number of clusters is much larger than the initial number of clusters, consider joining.


    def mutate(self):
        """
        Randomly join clusters:
        Since the recombination operator tends to lead to a larger number of clusters with mutation we join some of those.
        Join some clusters with probability
        """
        for ind in range(self.n_solutions):
            cluster_lengths = [len(clust) for clust in self.children[ind].clusters]
            n_clust = len(cluster_lengths)
            join_prob = self.avg_joins/n_clust # 2/Number of clusters: On average they get joined
            # Iterate through all clusters and select which ones have to be joined
            to_join = list(np.random.binomial(1, join_prob, size=n_clust)) # Select which entries have to be joined
            to_join_idx = [index for index, value in enumerate(to_join) if value != 0]
            # print(f"Vector:{to_join}\nPositions:{to_join_idx}")
            
            if len(to_join_idx) > 1:
                new_clust = []
                # print(f"clusters before:\n{self.children[ind].clusters}")
                for idx in sorted(to_join_idx, reverse=True):
                    new_clust += self.children[ind].clusters[idx]
                    del self.children[ind].clusters[idx]
                self.children[ind].clusters.append(new_clust)
                # print(f"clusters after:\n{self.children[ind].clusters}")
    
    def construct_children_splex(self):
        """
        Constructs the s-plexes based on the children assignment of counts.
        The s-plexes are built using the same heuristic as before.
        """
        for ind in range(self.n_solutions):
            self.children[ind].construct_all_splex()
            # print(f"Valid: {self.children[ind].check()}\nValue: {self.children[ind].calc_objective()}")

    def replace(self, perc_replace = 0.8):
        """
        - p: what percentage of the population should be replaced.
        Replaces the worst 80% of the initial population with the best 80% of the children population and sorts it again.
        Assumes that self.population and self.children are sorted and sorts the final population
        """
        new_population = self.population[0:int((1-perc_replace)*len(self.population))]
        new_population += self.children[int((1-perc_replace)*len(self.population)):]
        self.population = new_population
        self.evaluate_population()
        self.sort_population()
    
    def evolve_1_step(self):
        """
        Assuming the initial population has been generated, it performs the selection->recombination->mutation steps
        """
        self.recombine()
        self.mutate()
        self.construct_children_splex()
        self.evaluate_children()
        self.sort_children()
        self.replace()
    
    def evolve_n_steps(self, n):
        start_time = time.time()
        for i in range(n):
            print(f"Iteration: {i}")
            self.evolve_1_step()
            current_time = time.time() - start_time
            if current_time >= 15 * 60:
                print(f"Exceeded time limit")
                break
        total_time = time.time() - start_time
        print(f"Time for {n} steps: {total_time}s")
        print(f"Average time per step: {total_time/n}s")

          

    def __str__(self):
        return f"""
        Instance: {self.problem_instance.instance_type + '/' + self.problem_instance.problem_instance}
        Number of solutions: {self.n_solutions}
        (alpha, beta, k): {(self.alpha, self.beta, self.k)}
        Selection method: {self.selection_method}
        Replacement percentage: {self.perc_replace*100}%
        Average_joins: {self.avg_joins}
        """

if __name__ == '__main__':
    parser = get_settings_parser()
    parser.add_argument("inputfile", type=str, help='instance file path')
    parser.add_argument("-k", type=int, default=5, help='construction initial cluster size')
    parser.add_argument("--alpha", type=float, default=0.4, help='randomization for cluster initialization.')
    parser.add_argument("--beta", type=float, default=1, help='randomization for cluster assignment.')
    parser.add_argument("--iterations", type=int, default=5, help='iterations, for GA algorithm.')
    parser.add_argument("--n_solutions", type=int, default=5, help='number of solutions to be included in the population.')
    parser.add_argument("--perc_replace", type=float, default = 0.8, help = 'percentage of solutions that should be replaced.')
    parser.add_argument("--selection_method", type=str, default='fp', help='selection method for GA. lr: linear ranking, fp: fitness proportional selection.')
    parser.add_argument("--avg_joins", type=int, default=1, help='Average number of clusters that will be joined during mutation')

    args = parser.parse_args()
    spi = SPlexInstance(args.inputfile)
    GA_instance = GeneticAlgorithm(problem_instance=spi, n_solutions=args.n_solutions, k=args.k, alpha=args.alpha, beta=args.beta, selection_method=args.selection_method, perc_replace=args.perc_replace, avg_joins=args.avg_joins)
    print(GA_instance)
    GA_instance.generate_initial_population()
    GA_instance.initial_n_clusters
    """print(f"Top 10 Initial values:\n{GA_instance.population_values[0:10]}\n")
    GA_instance.evolve_n_steps(n=args.iterations)
    print(f"Top 10 Final values:\n{GA_instance.population_values[0:10]}\n")
    final_pop_clusters_number = [len(GA_instance.population[i].clusters) for i in range(GA_instance.n_solutions)]
    print(f"The final solutions have clusters of these size:\n{final_pop_clusters_number}")"""


    
    
   
# python GA.py 'data/inst_competition/heur049_n_300_m_17695.txt' -k 60 --iterations 5 --n_solutions 20 --selection_method 'lr' --perc_replace 0.8 --avg_joins 3
    





