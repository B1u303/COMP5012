import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

#load city data
def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    city_coords = {}
    for i, line in enumerate(lines):
        values = line.split()
        if len(values) >= 4:
            #temporarily using own index for cities
            city_coords[i] = (float(values[1]), float(values[2]), int(values[3]))  # (x, y, frequency)

    return city_coords

#euclidean distance
def euclidean_distance(city1, city2):
    x1, y1 = city1[:2]
    x2, y2 = city2[:2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#total tour distance considering the frequency of visits
def calculate_total_distance(tour, city_coords):
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities): 
        city_0 = city_coords[tour[i]]
        city_1 = city_coords[tour[(i+1) % num_cities]]
        #multiply the edge distance by the frequency of the originating city
        total_distance += euclidean_distance(city_0, city_1) * city_0[2]
    return total_distance

#longest segment (weighted by frequency).
def calculate_max_segment_distance(tour, city_coords):
    segment_distances = []
    num_cities = len(tour)
    for i in range(num_cities):
        city_0 = city_coords[tour[i]]
        city_1 = city_coords[tour[(i+1) % num_cities]]
        segment_distances.append(euclidean_distance(city_0, city_1) * city_0[2])
    return max(segment_distances)

#random initial tour
def generate_initial_solution(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

#non-dominated sorting on the population based on objectives
def non_dominated_sort(population, objectives):
    fronts = []
    dominated = [set() for _ in range(len(population))]
    domination_count = [0] * len(population)
    first_front = []

    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
            #check if solution i dominates solution j
            if (objectives[i][0] <= objectives[j][0] and objectives[i][1] <= objectives[j][1] and 
                (objectives[i][0] < objectives[j][0] or objectives[i][1] < objectives[j][1])):
                dominated[i].add(j)
            elif (objectives[j][0] <= objectives[i][0] and objectives[j][1] <= objectives[i][1] and 
                  (objectives[j][0] < objectives[i][0] or objectives[j][1] < objectives[i][1])):
                domination_count[i] += 1
        if domination_count[i] == 0:
            first_front.append(i)
    fronts.append(first_front)

    while first_front:
        next_front = []
        for i in first_front:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        first_front = next_front
        if first_front:
            fronts.append(first_front)
    return fronts

#assign crowding distance to each individual in the population
def crowding_distance_assignment(population, fronts, objectives):
    crowding_distances = [0] * len(population)
    for front in fronts:
        if len(front) <= 2:
            for idx in front:
                crowding_distances[idx] = float('inf')
            continue

        for obj_idx in range(2):
            sorted_indices = sorted(front, key=lambda x: objectives[x][obj_idx])
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')
            obj_min = objectives[sorted_indices[0]][obj_idx]
            obj_max = objectives[sorted_indices[-1]][obj_idx]
            if obj_max == obj_min:
                continue
            for i in range(1, len(sorted_indices) - 1):
                crowding_distances[sorted_indices[i]] += (
                    (objectives[sorted_indices[i+1]][obj_idx] - objectives[sorted_indices[i-1]][obj_idx]) /
                    (obj_max - obj_min)
                )
    return crowding_distances

#tounrament selection based on front ranking and crowding distance
def tournament_selection(population, crowding_distances, fronts):
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        front_rank_i = next(r for r, front in enumerate(fronts) if i in front)
        front_rank_j = next(r for r, front in enumerate(fronts) if j in front)
        if front_rank_i < front_rank_j:
            selected.append(population[i])
        elif front_rank_i > front_rank_j:
            selected.append(population[j])
        else:
            selected.append(population[i] if crowding_distances[i] > crowding_distances[j] else population[j])
    return selected

#order crossover
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end+1] = parent1[start:end+1]
    current_position = 0
    for city in parent2:
        if city not in child:
            while child[current_position] is not None:
                current_position += 1
            child[current_position] = city
    return child

#swap mutation 
def swap_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

#plot pareto front
def plot_pareto_front(objectives):
    plt.figure(figsize=(12, 7))
    plt.scatter([obj[0] for obj in objectives], [obj[1] for obj in objectives], 
                alpha=0.7, c='blue', label='Solutions')
    #convex hull for better readability 
    points = np.array([(obj[0], obj[1]) for obj in objectives])
    if len(points) > 2:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=2)
    plt.xlabel('Total Tour Distance (weighted by frequency)')
    plt.ylabel('Longest Segment Distance (weighted by frequency)')
    plt.title('Enhanced Pareto Front Visualization for PTSP')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#plot tour (2d)
def plot_tour(tour, cities, title):
    x = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    y = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", linestyle="-", color="b", label="Path")
    for i, (x_c, y_c, freq) in enumerate(cities.values()):
        plt.text(x_c, y_c, f"{i}\n({freq})", fontsize=10, color="red")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def moga_ptsp(file_path, population_size=50, generations=200, mutation_rate=0.1):
    #load cities
    cities = load_data(file_path)
    num_cities = len(cities)
    
    #initialize population 
    population = [generate_initial_solution(num_cities) for _ in range(population_size)]

    pareto_solutions = []
    pareto_objectives = []
    
    for gen in range(generations):
        print(f"Generation {gen + 1}")
        
        objectives = [
            (
                calculate_total_distance(tour, cities),
                calculate_max_segment_distance(tour, cities)
            )
            for tour in population
        ]
        
        #non-dominated sorting
        fronts = non_dominated_sort(population, objectives)
        
        #crowding distances
        crowding_distances = crowding_distance_assignment(population, fronts, objectives)
        
        #save the first front solutions
        first_front = fronts[0]
        pareto_solutions.extend([population[i] for i in first_front])
        pareto_objectives.extend([objectives[i] for i in first_front])
        
        #tournament selection
        selected_parents = tournament_selection(population, crowding_distances, fronts)
        
        #generate new population using order crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            p1, p2 = random.sample(selected_parents, 2)
            child1 = order_crossover(p1, p2)
            child2 = order_crossover(p2, p1)
            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = swap_mutation(child2)
            new_population.extend([child1, child2])
        population = new_population[:population_size]
    
    #plot pareto front for all pareto solutions
    plot_pareto_front(pareto_objectives)
    
    unique_objectives = []
    unique_indices = []
    for idx, obj in enumerate(pareto_objectives):
        if obj not in unique_objectives:
            unique_objectives.append(obj)
            unique_indices.append(idx)
    for i, idx in enumerate(unique_indices[:3]):
        dist, max_seg = pareto_objectives[idx]
        best_tour = pareto_solutions[idx]
        plot_tour(best_tour, cities, f'Tour - Total Dist: {dist:.2f}, Max Segment: {max_seg:.2f}')
    
    return pareto_solutions, pareto_objectives

if __name__ == "__main__":
    file_path = "D:/PTSP_Project/data/vrp8.txt"  
    best_solutions, best_objectives = moga_ptsp(file_path, population_size=50, generations=200, mutation_rate=0.1)
