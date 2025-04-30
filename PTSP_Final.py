import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# GA parameters
POP_SIZE = 400
NUM_GENERATIONS = 500
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2

#load cities from text file
def load_data(filename):
    city_coords = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 4:
                _, x, y, freq = parts
                city_coords.append((float(x), float(y), int(freq)))
    return city_coords

CITY_COORDINATES = load_data(
    r'D:/PTSP_Project/data/vrp8.txt'
)
NUM_CITIES = len(CITY_COORDINATES)


#calculate euclidean distance
def euclidean_distance(city1, city2):
    return np.linalg.norm(np.array(city1[:2]) - np.array(city2[:2]))

#create period visits based on frequencies
def create_period_visits(city_coords, num_periods=3):
    period_visits = [[] for _ in range(num_periods)]
    
    for city_idx, city in enumerate(city_coords):
        freq = city[2]  #get visit frequency of the city
        
        freq = max(1, min(freq, num_periods))
        
        #find which periods this city should be visited in
        periods_to_visit = []
        if freq == num_periods:
            #visit in all periods
            periods_to_visit = list(range(num_periods))
        else:
            #distribute visits evenly
            step = num_periods / freq
            for i in range(freq):
                period = int(i * step)
                periods_to_visit.append(period)
        
        #add city to the appropriate periods
        for period in periods_to_visit:
            period_visits[period].append(city_idx)
    
    return period_visits

#initialize population with period-specific tours
def initialize_population(pop_size, period_visits):
    population = []
    for _ in range(pop_size):
        #for each individual, create period-specific tours
        period_tours = []
        for cities in period_visits:
            period_tours.append(random.sample(cities, len(cities)))
        population.append(period_tours)
    return population

#total tour distance across all periods
def calculate_total_distance(period_tours, city_coords):
    total_distance = 0
    for tour in period_tours:
        if not tour:
            continue
            
        period_distance = 0
        for i in range(len(tour)):
            from_city = city_coords[tour[i]]
            to_city = city_coords[tour[(i + 1) % len(tour)]]
            period_distance += euclidean_distance(from_city, to_city)
        
        total_distance += period_distance
    
    return total_distance

#calculate workload balance across segments
def calculate_workload_balance(period_tours, city_coords):
    workloads = []
    
    for tour in period_tours:
        if not tour: 
            workloads.append(0)
            continue

        workload = 0
        for i in range(len(tour)):
            from_city = city_coords[tour[i]]
            to_city = city_coords[tour[(i + 1) % len(tour)]]
            workload += euclidean_distance(from_city, to_city)
        
        workloads.append(workload)
    
    return np.std(workloads)

#tournament selection based on front ranking and crowding distance
def tournament_selection(population, objectives, k):
    selected = random.sample(list(zip(population, objectives)), k)
    selected.sort(key=lambda x: (x[1][0], x[1][1]))
    return selected[0][0]

#order crossover
def ordered_crossover_periods(parent1, parent2):
    child = []
    
    for i in range(len(parent1)):
        p1_tour = parent1[i]
        p2_tour = parent2[i]
        
        if not p1_tour or not p2_tour:  
            child.append([])
            continue
            
        start, end = sorted(random.sample(range(len(p1_tour)), 2))
        period_child = [None] * len(p1_tour)
        period_child[start:end] = p1_tour[start:end]
        
        pointer = 0
        for city in p2_tour:
            if city not in period_child:
                while pointer < len(period_child) and period_child[pointer] is not None:
                    pointer += 1
                if pointer < len(period_child):
                    period_child[pointer] = city
        
        child.append(period_child)
    
    return child

#swao mutation for period-specific tours
def swap_mutation_periods(period_tours):
    mutated_tours = [tour[:] for tour in period_tours]
    
    non_empty_periods = [i for i, tour in enumerate(period_tours) if len(tour) >= 2]
    if not non_empty_periods:
        return mutated_tours
        
    period_idx = random.choice(non_empty_periods)
    tour = mutated_tours[period_idx]
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    
    return mutated_tours

#fast Non-Dominated Sorting (NSGA-II)
def fast_non_dominated_sort(objectives):
    S = [[] for _ in range(len(objectives))]
    front = [[]]
    n = [0] * len(objectives)
    rank = [0] * len(objectives)

    for p in range(len(objectives)):
        for q in range(len(objectives)):
            if p != q:
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

#pareto archive 
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
    
    # if archive exceeds max size, use crowding distance to trim
    if len(updated_archive_solutions) > max_archive_size:
        crowding_distances = calculate_crowding_distance(updated_archive_objectives)
        sorted_indices = np.argsort(crowding_distances)[::-1]
        updated_archive_solutions = [updated_archive_solutions[i] for i in sorted_indices[:max_archive_size]]
        updated_archive_objectives = [updated_archive_objectives[i] for i in sorted_indices[:max_archive_size]]
    
    return updated_archive_solutions, updated_archive_objectives

#crowding distance assignment (NSGA-II)
def calculate_crowding_distance(objectives):
    n = len(objectives)
    if n <= 2:
        return [float('inf')] * n
    
    crowding_distance = [0.0] * n
    
    for m in range(2):  
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

#find best solutions (objective specific) from the Pareto front
def find_special_solutions(archive_solutions, archive_objectives):
    if not archive_objectives:
        return [], []
    
    objectives_array = np.array(archive_objectives)
    
    #best distance solution (minimum distance)
    best_distance_idx = np.argmin(objectives_array[:, 0])
    
    #best balance solution (minimum workload standard deviation)
    best_balance_idx = np.argmin(objectives_array[:, 1])
    
    #compromise solution (closest to ideal point)
    min_vals = np.min(objectives_array, axis=0)
    max_vals = np.max(objectives_array, axis=0)
    range_vals = max_vals - min_vals
    normalized = (objectives_array - min_vals) / range_vals
    
    distances_to_ideal = np.sqrt(np.sum(normalized**2, axis=1))
    compromise_idx = np.argmin(distances_to_ideal)
    
    special_indices = [best_distance_idx, best_balance_idx, compromise_idx]
    special_solutions = [archive_solutions[idx] for idx in special_indices]
    special_objectives = [archive_objectives[idx] for idx in special_indices]
    
    return special_solutions, special_objectives

#plot pareto front
def plot_pareto_front(objectives, highlight_objectives=None, highlight_labels=None):
    distances, balances = zip(*objectives)
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, balances, c='blue', s=30, alpha=0.7, label='Pareto Front')

    if highlight_objectives and highlight_labels:
        for obj, label in zip(highlight_objectives, highlight_labels):
            plt.scatter(obj[0], obj[1], c='red', s=100, edgecolor='black', label=label)
    
    plt.xlabel('Total Distance')
    plt.ylabel('Workload Std Dev')
    plt.title('Pareto Front with Best Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()

#plotting period tours
def plot_period_tours(period_tours, city_coords, title='Period Tours'):
    plt.figure(figsize=(12, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for i, tour in enumerate(period_tours):
        if not tour: 
            continue
        color = colors[i % len(colors)]
        x = [city_coords[city][0] for city in tour + [tour[0]]]
        y = [city_coords[city][1] for city in tour + [tour[0]]]
        plt.plot(x, y, marker='o', color=color, label=f'Day {i+1}')
        for city in tour:
            plt.annotate(f"{city}({city_coords[city][2]})", 
                         (city_coords[city][0], city_coords[city][1]),
                         xytext=(5, 5), textcoords='offset points')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

#visualize crowding distance
def visualise_crowding_distance(objectives, solution_index=None, max_solutions=20):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #sample if too many solutions
    if len(objectives) > max_solutions:
        indices = np.linspace(0, len(objectives) - 1, max_solutions, dtype=int)
        objectives = [objectives[i] for i in indices]
    
    #sort by first objective (distance)
    sorted_idx = np.argsort([obj[0] for obj in objectives])
    sorted_obj = [objectives[i] for i in sorted_idx]
  
    crowding_distances = calculate_crowding_distance(sorted_obj)
    
    #choose a solution to highlight in plot
    if solution_index is None:
        #find a non-boundary solution with a good crowding distance
        mid_indices = [i for i in range(1, len(sorted_obj)-1)]
        if mid_indices:
            solution_index = mid_indices[len(mid_indices) // 2]
        else:
            solution_index = 0
    
    #plot
    plt.figure(figsize=(8, 6))
    
    x_vals = [obj[0] for obj in sorted_obj]
    y_vals = [obj[1] for obj in sorted_obj]
    
    #plot Pareto front
    plt.plot(x_vals, y_vals, 'b-', alpha=0.5)

    for i, (x, y, cd) in enumerate(zip(x_vals, y_vals, crowding_distances)):
        if cd == float('inf'):
            #boundary solutions
            plt.scatter(x, y, s=70, c='red', alpha=0.7)
        else:
            plt.scatter(x, y, s=50, c='blue', alpha=0.7)
 
    plt.scatter(
        sorted_obj[solution_index][0],
        sorted_obj[solution_index][1],
        s=100, c='green', edgecolor='black', zorder=10
    )
    
    #show crowding distance for the selected solution
    if solution_index > 0 and solution_index < len(sorted_obj) - 1:
        x = sorted_obj[solution_index][0]
        y = sorted_obj[solution_index][1]
        cd = crowding_distances[solution_index]
        
        left_x = sorted_obj[solution_index-1][0]
        right_x = sorted_obj[solution_index+1][0]
        
        #sort by second objective 
        y_sorted_idx = np.argsort([obj[1] for obj in sorted_obj])
        y_location = np.where(y_sorted_idx == solution_index)[0][0]
        
        if y_location > 0 and y_location < len(sorted_obj) - 1:
            lower_y = sorted_obj[y_sorted_idx[y_location-1]][1]
            upper_y = sorted_obj[y_sorted_idx[y_location+1]][1]
            
            plt.hlines(y, left_x, right_x, colors='red', linestyles='dashed', label='_nolegend_')
            plt.vlines(x, y - (y - lower_y)/2, y + (upper_y - y)/2, 
                      colors='red', linestyles='dashed', label='_nolegend_')
        else:
            plt.hlines(y, left_x, right_x, colors='red', linestyles='dashed', label='_nolegend_')
        
        #add label with crowding distance value
        plt.annotate(f"CD = {cd:.4f}", 
                    xy=(x, y),
                    xytext=(x + 20, y + 2),
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    
    # legend
    plt.scatter([], [], s=70, c='red', label='Boundary (CD = ∞)')
    plt.scatter([], [], s=50, c='blue', label='Regular solutions')
    plt.scatter([], [], s=100, c='green', edgecolor='black', label='Selected')
    
    plt.title('Crowding Distance Visualisation')
    plt.xlabel('Total Distance')
    plt.ylabel('Workload Std Dev')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return crowding_distances

#visualise crowding distance in the archive
def visualise_crowding_distance_in_archive(archive_objectives, specific_solution_idx=None):
    print("Visualising crowding distance for the Pareto front...")
    
    sorted_indices = np.argsort([obj[0] for obj in archive_objectives])
    sorted_objectives = [archive_objectives[i] for i in sorted_indices]
    
    #calculate crowding distances
    crowding_distances = calculate_crowding_distance(sorted_objectives)
    
    finite_distances = [cd for cd in crowding_distances if cd != float('inf')]
    if finite_distances:
        avg_cd = np.mean(finite_distances)
        print(f"Average Crowding Distance: {avg_cd:.4f}")
        print(f"Solutions with infinite CD: {len(crowding_distances) - len(finite_distances)}")
    
    visualise_crowding_distance(sorted_objectives, solution_index=specific_solution_idx)
    
    return crowding_distances

def moga_ptsp():
    num_periods = 5  
    period_visits = create_period_visits(CITY_COORDINATES, num_periods)
    
    #initialize population with period-specific tours
    population = initialize_population(POP_SIZE, period_visits)
    
    archive_solutions = []
    archive_objectives = []

    for gen in range(NUM_GENERATIONS):
        objectives = [(calculate_total_distance(tours, CITY_COORDINATES),
                       calculate_workload_balance(tours, CITY_COORDINATES)) 
                      for tours in population]

        fronts = fast_non_dominated_sort(objectives)

        #update archive 
        for idx in fronts[0]:
            archive_solutions, archive_objectives = update_archive(
                archive_solutions, 
                archive_objectives, 
                population[idx], 
                objectives[idx]
            )

        new_population = []
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, objectives, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, objectives, TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_RATE:
                child = ordered_crossover_periods(parent1, parent2)
            else:
                child = [tour[:] for tour in parent1]  # Deep copy

            if random.random() < MUTATION_RATE:
                child = swap_mutation_periods(child)

            new_population.append(child)

        population = new_population
        print(f"Generation {gen + 1} complete, Archive size: {len(archive_solutions)}")
        
        #only visualise at first generation and end to avoid slow execution
        if gen == 0 or gen == NUM_GENERATIONS - 1:
            print(f"\nVisualizing crowding distance at generation {gen + 1}:")
            visualise_crowding_distance_in_archive(archive_objectives)

    special_solutions, special_objectives = find_special_solutions(archive_solutions, archive_objectives)
    
    #plot pareto front with selected solutions
    plot_pareto_front(
        archive_objectives, 
        special_objectives, 
        ["Best Distance", "Best Balance", "Compromise"]
    )

    #plot selected solutions
    solution_titles = [
        f'Best Distance Solution - Total Dist: {special_objectives[0][0]:.2f}, Workload Std: {special_objectives[0][1]:.2f}',
        f'Best Balance Solution - Total Dist: {special_objectives[1][0]:.2f}, Workload Std: {special_objectives[1][1]:.2f}',
        f'Compromise Solution - Total Dist: {special_objectives[2][0]:.2f}, Workload Std: {special_objectives[2][1]:.2f}'
    ]
    
    for i, (solution, title) in enumerate(zip(special_solutions, solution_titles)):
        plot_period_tours(solution, CITY_COORDINATES, title)
    
    #final visualisation of crowding distance with all solutions
    print("\nFinal crowding distance visualization for the entire Pareto front:")
    visualise_crowding_distance_in_archive(archive_objectives)

if __name__ == '__main__':
    moga_ptsp()
