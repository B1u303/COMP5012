import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# GA parameters
POP_SIZE = 80
NUM_GENERATIONS = 100
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1

#load cities from text file
def load_data(file_path):
    city_coords = {}
    with open(file_path, "r") as f:
        lines = f.readlines() 

    for i, line in enumerate(lines):
        values = line.split()
        if len(values) >= 4:
            #temporarily using own index for cities
            city_coords[i] = (float(values[1]), float(values[2]), int(values[3]))
    
    return city_coords

#calculate euclidean distance
def euclidean_distance(city1, city2):
    x1, y1 = city1[:2]  
    x2, y2 = city2[:2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#total tour distance considering the frequency of visits
def calculate_total_distance(tour, city_coords):
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities): 
        city_0 = city_coords[tour[i]] 
        city_1 = city_coords[tour[(i+1) % num_cities]] 
        #removed frequency weighting 
        total_distance += euclidean_distance(city_0, city_1)
    
    return total_distance

#calculate workload balance across segments
def calculate_workload_balance(tour, city_coords, num_periods=3):
    segment_length = len(tour) // num_periods
    workloads = []

    for i in range(num_periods):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_periods - 1 else len(tour)
        segment = tour[start:end]

        workload = 0
        for j in range(len(segment) - 1):
            city_0 = city_coords[segment[j]]
            city_1 = city_coords[segment[j+1]]
            workload += euclidean_distance(city_0, city_1)
        workloads.append(workload)

    return np.std(workloads)

#random initial tour
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

#tournament selection based on front ranking and crowding distance
def tournament_selection(population, objectives, k):
    selected = random.sample(list(zip(population, objectives)), k)
    selected.sort(key=lambda x: (x[1][0], x[1][1]))
    return selected[0][0]

#order crossover
def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    pointer = 0
    for city in parent2:
        if city not in child:
            while pointer < len(child) and child[pointer] is not None:
                pointer += 1
            if pointer < len(child):
                child[pointer] = city
    return child

#swap mutation
def swap_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

#fast Non-Dominated Sorting (NSGA-II)
def fast_non_dominated_sort(objectives):
    S = [[] for _ in range(len(objectives))]
    front = [[]]
    n = [0] * len(objectives)
    rank = [0] * len(objectives)

    for p in range(len(objectives)):
        for q in range(len(objectives)):
            if (objectives[p][0] < objectives[q][0] and objectives[p][1] < objectives[q][1]) or \
               (objectives[p][0] <= objectives[q][0] and objectives[p][1] < objectives[q][1]) or \
               (objectives[p][0] < objectives[q][0] and objectives[p][1] <= objectives[q][1]):
                S[p].append(q)
            elif (objectives[q][0] < objectives[p][0] and objectives[q][1] < objectives[p][1]) or \
                 (objectives[q][0] <= objectives[p][0] and objectives[q][1] < objectives[p][1]) or \
                 (objectives[q][0] < objectives[p][0] and objectives[q][1] <= objectives[p][1]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)

    return front[:-1]

#check if solution dominates another
def dominates(obj1, obj2):
    return (obj1[0] <= obj2[0] and obj1[1] < obj2[1]) or \
           (obj1[0] < obj2[0] and obj1[1] <= obj2[1])

#crowding distance assignment (NSGA-II)
def calculate_crowding_distance(objectives):
    n = len(objectives)
    if n <= 2:
        return [float('inf')] * n
    
    crowding_distance = [0.0] * n
    
    for m in range(2):  #two objectives
        sorted_indices = np.argsort([obj[m] for obj in objectives])

        crowding_distance[sorted_indices[0]] = float('inf')
        crowding_distance[sorted_indices[-1]] = float('inf')

        obj_min = objectives[sorted_indices[0]][m]
        obj_max = objectives[sorted_indices[-1]][m]

        if obj_max > obj_min:  
            for i in range(1, n-1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                next_idx = sorted_indices[i+1]
                
                crowding_distance[idx] += (objectives[next_idx][m] - objectives[prev_idx][m]) / (obj_max - obj_min)
    
    return crowding_distance

#pareto archive (non-dominated solutions)
def update_archive(archive_solutions, archive_objectives, new_solution, new_objective, max_archive_size=100):
    #check if new solution is dominated by any archive solution
    for archive_obj in archive_objectives:
        if dominates(archive_obj, new_objective):
            return archive_solutions, archive_objectives  

    #remove solutions from archive that are dominated by new solution
    non_dominated_indices = []
    for i, archive_obj in enumerate(archive_objectives):
        if not dominates(new_objective, archive_obj):
            non_dominated_indices.append(i)
    
    updated_archive_solutions = [archive_solutions[i] for i in non_dominated_indices]
    updated_archive_objectives = [archive_objectives[i] for i in non_dominated_indices]

    updated_archive_solutions.append(new_solution)
    updated_archive_objectives.append(new_objective)
    
    #if archive exceeds max size, use crowding distance to trim
    if len(updated_archive_solutions) > max_archive_size:
        crowding_distances = calculate_crowding_distance(updated_archive_objectives)
        sorted_indices = np.argsort(crowding_distances)[::-1]
        updated_archive_solutions = [updated_archive_solutions[i] for i in sorted_indices[:max_archive_size]]
        updated_archive_objectives = [updated_archive_objectives[i] for i in sorted_indices[:max_archive_size]]
    
    return updated_archive_solutions, updated_archive_objectives

#plot pareto front
def plot_pareto_front(objectives):
    plt.figure(figsize=(10, 6))
    distances, balances = zip(*objectives)
    plt.scatter(distances, balances, c='blue', alpha=0.7, label='Pareto Front')
    
    #convex hull for better readability
    points = np.array([(obj[0], obj[1]) for obj in objectives])
    if len(points) > 2:  
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=2)
            
    plt.xlabel('Total Tour Distance')
    plt.ylabel('Workload Balance (Std Dev)')
    plt.title('Pareto Front for PTSP')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

#plot tour (2d)
def plot_tour(tour, city_coords, title):
    x = [city_coords[city][0] for city in tour + [tour[0]]]
    y = [city_coords[city][1] for city in tour + [tour[0]]]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", linestyle="-", color="b")
    for i, city in enumerate(tour):
        x_c, y_c, freq = city_coords[city]
        plt.text(x_c, y_c, f"{city}\n({freq})", fontsize=10, color="red")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.grid(True)
    plt.show()

def moga_ptsp(file_path, population_size=POP_SIZE, generations=NUM_GENERATIONS, mutation_rate=MUTATION_RATE):
    #load cities
    city_coords = load_data(file_path)
    num_cities = len(city_coords)
    
    #initialize population
    population = initialize_population(population_size, num_cities)
    archive_solutions = []
    archive_objectives = []

    for gen in range(generations):
        print(f"Generation {gen + 1}")
  
        objectives = []
        for tour in population:
            obj1 = calculate_total_distance(tour, city_coords)
            obj2 = calculate_workload_balance(tour, city_coords)
            objectives.append((obj1, obj2))  

        #non-dominated sorting
        fronts = fast_non_dominated_sort(objectives)
        
        #update archive with solutions from the first front
        for idx in fronts[0]:
            archive_solutions, archive_objectives = update_archive(
                archive_solutions, 
                archive_objectives, 
                population[idx], 
                objectives[idx]
            )
        
        #tournament selection and create new population
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, objectives, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, objectives, TOURNAMENT_SIZE)
            
            if random.random() < CROSSOVER_RATE:
                child = ordered_crossover(parent1, parent2)
            else:
                child = parent1[:]
                
            if random.random() < mutation_rate:
                child = swap_mutation(child)
                
            new_population.append(child)
            
        population = new_population
        print(f"Generation {gen + 1} complete, Archive size: {len(archive_solutions)}")
    
    # plot pareto front from archive
    plot_pareto_front(archive_objectives)
    
    unique_objectives = []
    unique_indices = []
    for idx, obj in enumerate(archive_objectives):
        if obj not in unique_objectives:
            unique_objectives.append(obj)
            unique_indices.append(idx)
 
    for i, idx in enumerate(unique_indices[:3]):  
        dist, balance = archive_objectives[idx]
        best_tour = archive_solutions[idx]
        plot_tour(best_tour, city_coords, f'Tour - Total Dist: {dist:.2f}, Workload Balance: {balance:.2f}')
    
    return archive_solutions, archive_objectives

if __name__ == "__main__":
    file_path = "D:/PTSP_Project/data/vrp8.txt"  
    archive_solutions, archive_objectives = moga_ptsp(file_path, population_size=POP_SIZE, generations=NUM_GENERATIONS, mutation_rate=MUTATION_RATE)
