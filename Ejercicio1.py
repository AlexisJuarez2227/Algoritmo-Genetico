import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from prettytable import PrettyTable

# Configuración y ejecución del algoritmo
start_value = float(input("Ingrese el valor inicial: "))
end_value = float(input("Ingrese el valor final: "))
precision = float(input("Ingrese el deltaX: "))
generations_count = int(input("Ingrese el número de generaciones: "))
maximize = input("¿Quieres maximizar la función? (s/n): ").lower() == 's'
mutation_prob_gene = float(input("Ingrese la probabilidad de mutación del gen: "))
mutation_prob_individual = float(input("Ingrese la probabilidad de mutación individual: "))
individuals_count = int(input("Ingrese el número de individuos: "))
max_population = int(input("Ingrese la población máxima: "))
crossover_rate = float(input("Ingrese la tasa de cruza: "))

# Calcular el número de bits necesarios
bit_length = math.ceil(math.log2((end_value - start_value) / precision + 1))

def float_to_binary(value, min_value, max_value):
    scaled_value = (value - min_value) / (max_value - min_value) * (2**bit_length - 1)
    return format(int(scaled_value), '0' + str(bit_length) + 'b')

def binary_to_float(binary_str, min_value, max_value):
    int_value = int(binary_str, 2)
    return min_value + int_value * (max_value - min_value) / (2**bit_length - 1)

def fitness_function(individual, maximize, min_value, max_value):
    x = binary_to_float(individual, min_value, max_value)
    f = x * np.cos(x)
    return f if maximize else -f

def create_initial_population(count, min_value, max_value):
    return [float_to_binary(random.uniform(min_value, max_value), min_value, max_value) for _ in range(count)]

def select_pairs(population):
    pairs = []
    n = len(population)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((population[i], population[j]))
    return pairs

def crossover(pair):
    crossover_point = random.randint(1, bit_length - 1)
    child1 = pair[0][:crossover_point] + pair[1][crossover_point:]
    child2 = pair[1][:crossover_point] + pair[0][crossover_point:]
    return child1, child2

def mutate(individual, mutation_prob_gene):
    individual = list(individual)
    for i in range(bit_length):
        if random.random() < mutation_prob_gene:
            individual[i] = '1' if individual[i] == '0' else '0'
    return ''.join(individual)

def prune(population, max_population, min_value, max_value, maximize):
    unique_population = list(set(population))
    unique_population.sort(key=lambda ind: fitness_function(ind, maximize, min_value, max_value), reverse=maximize)
    if len(unique_population) > max_population:
        best_individual = unique_population[0]
        to_keep = random.sample(unique_population[1:], max_population - 1)
        to_keep.append(best_individual)
        unique_population = to_keep
    statistics = {
        "max": fitness_function(unique_population[0], maximize, min_value, max_value),
        "min": fitness_function(unique_population[-1], maximize, min_value, max_value),
        "average": sum(fitness_function(ind, maximize, min_value, max_value) for ind in unique_population) / len(unique_population)
    }
    return unique_population, statistics

def plot_function_with_individuals(x_values, y_values, individuals, best, worst, generation, folder, min_value, max_value, maximize):
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='f(x) = x * cos(x)')

    x_individuals = [binary_to_float(ind, min_value, max_value) for ind in individuals]
    y_individuals = [x * np.cos(x) for x in x_individuals]

    plt.scatter(x_individuals, y_individuals, color='blue', label='Individuos', alpha=0.6)

    best_x = binary_to_float(best, min_value, max_value)
    best_y = best_x * np.cos(best_x)
    plt.scatter([best_x], [best_y], color='green', label='Mejor Individuo', s=100, edgecolor='black')

    worst_x = binary_to_float(worst, min_value, max_value)
    worst_y = worst_x * np.cos(worst_x)
    plt.scatter([worst_x], [worst_y], color='red', label='Peor Individuo', s=100, edgecolor='black')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Función y Individuos - Generación {generation}')
    plt.legend()
    plt.grid(True)

    plt.xlim(min_value, max_value)
    plt.ylim(min(y_values), max(y_values))

    plot_name = f"Generation_{generation}.png"
    plt.savefig(os.path.join(folder, plot_name))
    plt.close()

def plot_evolution(best_fitnesses, worst_fitnesses, average_fitnesses, folder, maximize):
    plt.figure(figsize=(10, 5))
    
    plt.plot(best_fitnesses, label='Mejor Aptitud', color='green')
    plt.plot(worst_fitnesses, label='Peor Aptitud', color='red')
    plt.plot(average_fitnesses, label='Aptitud Media', color='blue')

    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    if maximize:
        plt.title('Evolución de la Maximización de Aptitudes')
    else:
        plt.title('Evolución de la Minimización de Aptitudes')

    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(folder, 'Evolution_Fitness.png'))
    plt.close()

def genetic_algorithm(start_value, end_value, precision, generations_count, maximize, mutation_prob_gene, mutation_prob_individual, individuals_count, max_population):
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    min_value = start_value
    max_value = end_value

    x_values = np.linspace(min_value, max_value, 400)
    y_values = x_values * np.cos(x_values)

    population = create_initial_population(individuals_count, min_value, max_value)
    best_fitnesses = []
    worst_fitnesses = []
    average_fitnesses = []

    for generation in range(generations_count + 1):
        fitnesses = [fitness_function(ind, maximize, min_value, max_value) for ind in population]
        best_fitness = max(fitnesses) if maximize else min(fitnesses)
        worst_fitness = min(fitnesses) if maximize else max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)

        best_fitnesses.append(best_fitness)
        worst_fitnesses.append(worst_fitness)
        average_fitnesses.append(average_fitness)

        best_individual = population[fitnesses.index(best_fitness)]
        worst_individual = population[fitnesses.index(worst_fitness)]

        best_x_value = binary_to_float(best_individual, min_value, max_value)
        
        # Imprimir tabla de la generación actual
        best_table = PrettyTable()
        best_table.field_names = ["Generación", "Cadena de Bits", "Índice", "Valor de x", "Valor de Aptitud"]
        best_table.add_row([generation, best_individual, fitnesses.index(best_fitness), round(best_x_value, 3), round(best_fitness, 3)])
        print(best_table)

        print(f"Generación {generation}: Mejor = {round(best_fitness, 3)}, Peor = {round(worst_fitness, 3)}, Media = {round(average_fitness, 3)}")

        plot_function_with_individuals(
            x_values, y_values, population, best_individual, worst_individual, generation, plots_folder, min_value, max_value, maximize)

        if generation < generations_count:
            pairs = select_pairs(population)
            new_population = []

            for pair in pairs:
                if random.random() < crossover_rate:
                    offspring = crossover(pair)
                    new_population.extend(offspring)
                else:
                    new_population.extend(pair)

            new_population = [mutate(ind, mutation_prob_gene) for ind in new_population]

            # Validar los nuevos individuos
            population = [ind for ind in new_population if min_value <= binary_to_float(ind, min_value, max_value) <= max_value]

            # Poda intermedia para mantener la población dentro de los límites
            population, stats = prune(population, max_population, min_value, max_value, maximize)

            # Agregar el mejor individuo de la generación anterior (elitismo)
            population.append(best_individual)

    # Poda final
    population, stats = prune(population, max_population, min_value, max_value, maximize)

    plot_evolution(best_fitnesses, worst_fitnesses, average_fitnesses, plots_folder, maximize)
    create_video(plots_folder, generations_count)

def create_video(folder, generations_count):
    image_folder = folder
    video_name = 'GeneticAlgorithmVideo.avi'

    images = [f"Generation_{i}.png" for i in range(0, generations_count + 1)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

genetic_algorithm(
    start_value, end_value, precision, generations_count, maximize,
    mutation_prob_gene, mutation_prob_individual, individuals_count, max_population)
