def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    city_coords = {}
    for i, line in enumerate(lines):
        values = line.split()
        if len(values) >= 3:
            city_coords[i] = (float(values[1]), float(values[2]))

    return city_coords

#laod city coordinates 
file_path = "D:/PTSP_Project/data/vrp8.txt"
cities = load_data(file_path)
print("Loaded cities", cities)

import math

#finding euclidean dist
def euclidean_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)           #euclidean distance formula 

#eg test
#city_0 = (37.0, 52.0)
#city_1 = (49.0, 49.0)
#print("Euclidean Distance:", euclidean_distance(city_0, city_1))

def compute_distance_matrix(city_coords):
    num_cities = len(city_coords)
    distance_matrix = [[0] * num_cities for _ in range(num_cities)]

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(city_coords[i], city_coords[j])

    return distance_matrix
        
distance_matrix = compute_distance_matrix(cities)
print("Distance Matrix:", distance_matrix)

#generate a random initial tour 
import random

def generate_initial_solution(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

initial_tour = generate_initial_solution(len(cities))
print("InitiaL Tour:", initial_tour)

#calculate total distance of the tour 

def calcualte_total_distance(tour, city_coords):
    total_distance = 0
    num_cities = len(tour)

    for i in range(num_cities):
        city_0 = city_coords[tour[i]]
        city_1 = city_coords[tour[(i+1) % num_cities]]
        total_distance += euclidean_distance(city_0, city_1)

    return total_distance

#eg test

initial_distance = calcualte_total_distance(initial_tour, cities)
print("Initail tour distance:", initial_distance)