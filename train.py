import random
import argparse
import numpy as np
from ai import Game
from inits import *
import os
import imageio.v2 as imageio  # For the production of GIF


class Individual:
    """Individual in population of Genetic Algorithm.
    Attributes:
        genes: A list which can transform to weight of Neural Network.
        score: Score of the snake played by its Neural Network.
        steps: Steps of the snake played by its Neural Network.
        fitness: Fitness of Individual.
        seed: The random seed of the game, saved for reproduction.
    """

    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.fitness = 0
        self.seed = None

    def get_fitness(self):
        """Get the fitness of Individual."""
        game = Game([self.genes])
        self.score, self.steps, self.seed = game.play()
        # self.fitness = (self.score + 1 / self.steps) * 100000
        self.fitness = (.25 * self.steps + ((2 ** self.score) + (self.score ** 2.1) * 500) -
                        ((.25 * self.steps) ** 1.3)) * 100000  # fitness functions
        self.fitness = max(self.fitness, .1)


class GA:
    """Genetic Algorithm.遗传算法
    Attributes:
        p_size: Size of the parent generation.
        c_size: Size of the child generation.
        genes_len: Length of the genes.
        mutate_rate: Probability of the mutation.
        population: A list of individuals.
        best_individual: Individual with the best fitness.
        avg_score: Average score of the population.
    """

    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE):
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        self.best_individual = None
        self.avg_score = 0

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))

    def inherit_ancestor(self):
        """Load genes from './genes/parents/{i}', i: the ith individual."""
        for i in range(self.p_size):
            pth = os.path.join("genes", "parents", str(i))
            with open(pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                self.population.append(Individual(genes))

    def crossover(self, c1_genes, c2_genes, pc):
        """Single-point crossover."""
        point = np.random.randint(0, self.genes_len)
        po = random.random()  # 0-1
        if po < pc:
            c1_genes[:point + 1], c2_genes[:point + 1] = c2_genes[:point + 1], c1_genes[:point + 1]

    def TwoCrossover(self, c1_genes, c2_genes, pc):
        """Two-point crossover"""
        point1 = np.random.randint(0, self.genes_len)
        point2 = np.random.randint(0, self.genes_len)
        po = random.random()  # 0-1
        if po < pc:
            if point1 > point2 and point1 != point2:
                point1, point2 = point2, point1
            if point1 != point2:
                temp1 = c1_genes[point1:point2].copy()
                temp2 = c2_genes[point1:point2].copy()
                c1_genes[point1:point2] = temp2
                c2_genes[point1:point2] = temp1

    def uniform_binary_crossover(self, c1_genes, c2_genes):
        """Uniform-binary crossover"""
        offspring1 = c1_genes.copy()
        offspring2 = c2_genes.copy()

        mask = np.random.uniform(0, 1, size=offspring1.shape)
        offspring1[mask > 0.5] = c2_genes[mask > 0.5]
        offspring2[mask > 0.5] = c1_genes[mask > 0.5]

        return offspring1, offspring2

    def gaussian_mutation(self, c_genes,
                          scale,
                          mu=None,
                          sigma=None):
        """
        Perform a gaussian mutation for each gene in an individual with probability, prob_mutation.

        If mu and sigma are defined, then the gaussian distribution will be drawn from that,
        otherwise it will be drawn from N(0, 1) for the shape of the individual.
        """
        # Determine which genes will be mutated
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        # If mu and sigma are defined, create gaussian distribution around each one
        if mu is not None and sigma is not None:
            gaussian_mutation = np.random.normal(mu, sigma, size=c_genes.shape)
        # Otherwise center around N(0,1)
        else:
            gaussian_mutation = np.random.normal(size=c_genes.shape)

        if scale:
            gaussian_mutation[mutation_array] *= scale

        # Update
        c_genes[mutation_array] += gaussian_mutation[mutation_array]

    def elitism_selection(self, size):
        """Select the top #size individuals to be parents."""
        # Arrange from large to small
        population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def roulette_wheel_selection(self, size):  # roulette-wheel selection strategy
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        for _ in range(size):
            pick = np.random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break

        return selection

    def tournament_selection(self, num_individuals, tournament_size):  # Tournament Selection
        selection = []
        for _ in range(num_individuals):
            tournament = np.random.choice(self.population, tournament_size)
            best_from_tournament = max(tournament, key=lambda individual: individual.fitness)
            selection.append(best_from_tournament)

        return selection

    def evolve(self):
        """The main process of Genetic Algorithm."""
        sum_score = 0
        for individual in self.population:
            individual.get_fitness()
            sum_score += individual.score
        self.avg_score = sum_score / len(self.population)

        self.population = self.elitism_selection(self.p_size)  # Select parents to generate children.
        self.best_individual = self.population[0]
        random.shuffle(self.population)

        # Generate children.
        children = []
        while len(children) < self.c_size:
            p1, p2 = self.roulette_wheel_selection(2)
            # p1, p2 = self.tournament_selection(2, 10)
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()
            # self.uniform_binary_crossover(c1_genes, c2_genes)
            # self.crossover(c1_genes, c2_genes, 0.8)  # pc: crossover probability
            self.TwoCrossover(c1_genes, c2_genes, 0.8)
            self.gaussian_mutation(c1_genes, 0.2)
            self.gaussian_mutation(c2_genes, 0.2)
            c1, c2 = Individual(c1_genes), Individual(c2_genes)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)

    def save_best(self):
        """Save the best individual that can get #score so far."""
        score = self.best_individual.score
        genes_pth = os.path.join("genes", "best", str(score))
        os.makedirs(os.path.dirname(genes_pth), exist_ok=True)
        with open(genes_pth, "w") as f:
            for gene in self.best_individual.genes:
                f.write(str(gene) + " ")
        seed_pth = os.path.join("seed", str(score))
        os.makedirs(os.path.dirname(seed_pth), exist_ok=True)
        with open(seed_pth, "w") as f:
            f.write(str(self.best_individual.seed))

    def save_best_gif(self, gif_dir="gifs", gif_duration=0.08):
        """Save a gif of the best individual's game play."""
        score = self.best_individual.score
        genes = self.best_individual.genes
        seed = self.best_individual.seed
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"best_{score}.gif")
        from ai import Game
        game = Game(show=True, genes_list=[genes], seed=seed)
        # Use the new play_and_capture method
        game.play_and_capture(capture_frames=True, gif_path=gif_path, gif_duration=gif_duration)

    def save_all(self):
        """Save the top population."""
        for individual in self.population:
            individual.get_fitness()
        population = self.elitism_selection(self.p_size)
        for i in range(len(population)):
            pth = os.path.join("genes", "parents", str(i))
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            with open(pth, "w") as f:
                for gene in population[i].genes:
                    f.write(str(gene) + " ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--inherit', action="store_true",
                        help="whether to load genes from path ./genes/parents.")
    args = parser.parse_args()

    ga = GA()

    if args.inherit:
        ga.inherit_ancestor()
    else:
        ga.generate_ancestor()

    generation = 0  # frequency
    max_g = 1200  # max generation
    record = 0  # possible marks
    all_data = []  # using lists to store all data avoids possible bounds violations for fixed-size arrays

    # Display once every 40 generations.
    show_every_n_generations = 40

    while generation < max_g:
        generation += 1
        ga.evolve()
        print("generation:", generation, ", record:", record,
              ", best score:", ga.best_individual.score, ", average score:", ga.avg_score)
        all_data.append([generation, record, ga.best_individual.score, ga.avg_score])

        if ga.best_individual.score > record:
            record = ga.best_individual.score
            ga.save_best()
            ga.save_best_gif()  # Save gif when new record

        if generation % show_every_n_generations == 0:
            print(f"\n--- Displaying the individual's game at generation {generation} ---")
            try:
                genes = ga.best_individual.genes
                seed = ga.best_individual.seed
                game = Game(show=True, genes_list=[genes], seed=seed)
                game.play()
                print(f"--- Display finished, continue training ---\n")
            except Exception as e:
                print(f"Error occurred during display: {e}. Continue training...")
        elif generation == max_g:
            print(f"--- Display finished, stop training ---\n")

        if generation % show_every_n_generations == 0:
            ga.save_all()
            with open('generation_all.txt', 'a') as fw:
                for gen_data in all_data[-40:]:
                    row = ' '.join(map(str, gen_data)) + '\n'
                    fw.write(row)
            all_data = []

