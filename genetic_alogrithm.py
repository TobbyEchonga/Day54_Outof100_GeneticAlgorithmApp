import numpy as np

# Define the problem: Find a binary string that maximizes the number of ones.

# Genetic Algorithm Parameters
population_size = 10
chromosome_length = 10
mutation_rate = 0.1
generations = 100

def initialize_population(population_size, chromosome_length):
    return np.random.randint(2, size=(population_size, chromosome_length))

def fitness(individual):
    return np.sum(individual)

def selection(population, fitness_values):
    probabilities = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    mutation_mask = np.random.choice([0, 1], size=len(individual), p=[1 - mutation_rate, mutation_rate])
    return (individual + mutation_mask) % 2

def genetic_algorithm():
    population = initialize_population(population_size, chromosome_length)

    for generation in range(generations):
        fitness_values = np.array([fitness(individual) for individual in population])

        # Select parents based on fitness
        parents = selection(population, fitness_values)

        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = np.array(new_population)

        # Print best individual in each generation
        best_individual = population[np.argmax(fitness_values)]
        print(f"Generation {generation + 1}: Best Individual - {best_individual}, Fitness - {fitness(best_individual)}")

if __name__ == '__main__':
    genetic_algorithm()
