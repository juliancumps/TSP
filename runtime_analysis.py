
import math
import numpy as np
import itertools
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from matplotlib.widgets import Slider
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import heapq
#geodesics
import pygeodesic
import pygeodesic.geodesic as geodesic



#define city coords
random.seed(33)  #set seed for reproducibility
num_cities= 10    #choose number of cities
board_size= 50
cities = {i: (round(random.uniform(3, board_size -3), 2), round(random.uniform(3, board_size - 3), 2),round(random.uniform(0, 10), 2)) for i in range(num_cities)}


#generate the terrain points and plot
random.seed(43)
num_terrain_points=50
terrain_points={i: (round(random.uniform(0, board_size), 2), round(random.uniform(0, board_size), 2),round(random.uniform(0.2, 2), 2)) for i in range(num_terrain_points)}

# Add  corner points to the terrain
corner_points = {
    num_terrain_points: (0, 0, 0),
    num_terrain_points + 1: (board_size, 0, 0),
    num_terrain_points + 2: (0, board_size, 0),
    num_terrain_points + 3: (board_size, board_size, 0),
}
terrain_points.update(corner_points)

# Extract x and y coordinates for Delaunay triangulation
xy_points = np.array([(point[0], point[1]) for point in terrain_points.values()])

# Compute Delaunay triangulation
tri = Delaunay(xy_points)


# Helper function to calculate z-coordinate on a plane
def plane_z(x, y, triangle_points):
    """
    Compute the z-coordinate of a plane at (x, y) using three points on the plane.
    """
    p1, p2, p3 = triangle_points
    # Vectors defining the plane
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)

    # Normal vector to the plane
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = -np.dot(normal, p1)

    # Compute z using the plane equation ax + by + cz + d = 0
    if c == 0:  # Avoid division by zero (rare, but handle flat triangles)
        return p1[2]
    z = -(a * x + b * y + d) / c
    return z

# Update cities' z-coordinates
for i, (x, y, _) in cities.items():
    simplex = tri.find_simplex((x, y))
    if simplex != -1:  # Ensure the point lies within the triangulation
        triangle_indices = tri.simplices[simplex]
        triangle_points = [(*xy_points[idx], terrain_points[idx][2]) for idx in triangle_indices]
        cities[i] = (x, y, plane_z(x, y, triangle_points))

####################################        Make the terrain a closed polygon



#Create a combined list of cities and terrain points
combined_points = {}
for i, coords in terrain_points.items():
    combined_points[i] = coords

l = len(combined_points)
    
for i, coords in cities.items():
    combined_points[i+l] = coords



xy_combined_points = np.array([(point[0], point[1]) for point in combined_points.values()])



tri_2 = Delaunay(xy_combined_points)



tri_2.simplices = np.concatenate((tri_2.simplices, np.array([[l-4, l-3, l-1], [l-2, l-4, l-1]])), axis = 0)

#make vertices and faces lists for the geodesics library
# Convert faces to a numpy array
faces = [tuple(triangle) for triangle in tri_2.simplices]  # Already a list of tuples
faces = np.array(faces, dtype=np.int32)  # Use an integer data type for indices

# Convert vertices to a numpy array
vertices = list(combined_points.values())  # Already a list of coordinate lists/tuples

vertices = np.array(vertices, dtype=np.float64)  # Use a float data type for coordinates


geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

###############################################

#calculate distance between two cities
def distance(city1,city2):

   distance, path = geoalg.geodesicDistance(city2+l, city1+l)

   return distance, path

#find total tour distance
def total_distance(tour):
    total=0
    total_path = np.empty((0, 3)) 
    for i in range(len(tour)-1):
        d, path_1 = distance(tour[i],tour[i+1])
        total+= d
        total_path = np.concatenate((total_path, path_1), axis = 0)
    d, path_1 = distance(tour[-1],tour[0])
    total+= d #go back to start city
    total_path = np.concatenate((total_path, path_1), axis = 0)
    return total, total_path


#simulated annealing function!
def simulated_annealing(cities,initial_solution,temperature,cooling_rate):
    current_solution=initial_solution
    current_cost, path_to =total_distance(current_solution)
    best_solution = current_solution
    best_path = path_to
    best_cost=current_cost
    #main loop
    while temperature>0.1:
        #generate a random neighbor solution to start at
        neighbor_solution=current_solution[:]
        i,j=sorted(random.sample(range(len(neighbor_solution)),2))
        neighbor_solution[i:j+1]=reversed(neighbor_solution[i:j+1])
        #calculate the cost difference between the current and neighboring tours
        neighbor_cost, neighbor_path =total_distance(neighbor_solution)
        cost_difference=neighbor_cost-current_cost
        #if new solution is better, accept it and make currenbt
        if cost_difference<0:
            current_solution= neighbor_solution
            current_cost= neighbor_cost
            if neighbor_cost<best_cost:
                best_solution=neighbor_solution
                best_cost=neighbor_cost
                best_path = neighbor_path
        #if the new solution is worse, maybe accept it (decision based on temp)
        elif random.random()<math.exp(-cost_difference/temperature):
            current_solution=neighbor_solution
            current_cost=neighbor_cost
        #cool the temperature by the cooling rate
        temperature*=cooling_rate
    return best_solution,best_cost, best_path #return the best (smallest distance) tour and the second return value is the distance of this tour



#################################################################





###################################################################################


# Assign colors to triangles using the 4-color theorem
triangle_colors = {}
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33']  # Four distinct colors
adjacency_list = {i: set() for i in range(len(tri.simplices))}

# Build adjacency list for triangles
for i, simplex1 in enumerate(tri.simplices):
    for j, simplex2 in enumerate(tri.simplices):
        if i != j and len(set(simplex1) & set(simplex2)) == 2:  # Two shared vertices
            adjacency_list[i].add(j)

# Assign colors ensuring no adjacent triangles share the same color
for triangle, neighbors in adjacency_list.items():
    neighbor_colors = {triangle_colors[neighbor] for neighbor in neighbors if neighbor in triangle_colors}
    for color in colors:
        if color not in neighbor_colors:
            triangle_colors[triangle] = color
            break





#generate an initial solution (at random)
initial_solution=list(cities.keys())
random.shuffle(initial_solution)

#initialize stuff for simulated annealing function
initial_temperature=1000
cooling_rate=0.9 #reasonable time and accuracy

#simulate the anneal! store it in a variable so we can call it later
#optimal_tour,optimal_distance, optimal_path =simulated_annealing(cities, initial_solution,initial_temperature,cooling_rate)


# Function to generate random city coordinates
def generate_cities(num_cities, board_size=50):
    random.seed(random.randint(1,99))
    return {i: (round(random.uniform(3, board_size - 3), 2),
                round(random.uniform(3, board_size - 3), 2),
                round(random.uniform(0, 10), 2)) for i in range(num_cities)}
    
num_trials = 3
runtime_results = []
runtime_results_sqrt = []
n_list = [10,20,30,40,50,60,70,80,90,100]

print('computing....')
# Measure runtime for different numbers of cities
for num_cities in n_list:
    runtimes = []
    for _ in range(num_trials):
        cities = generate_cities(num_cities)        
        #generate the terrain points and plot
        random.seed(random.randint(1,99))
        num_terrain_points=50
        terrain_points={i: (round(random.uniform(0, board_size), 2), round(random.uniform(0, board_size), 2),round(random.uniform(0.2, 2), 2)) for i in range(num_terrain_points)}
        # Add  corner points to the terrain
        corner_points = {
            num_terrain_points: (0, 0, 0),
            num_terrain_points + 1: (board_size, 0, 0),
            num_terrain_points + 2: (0, board_size, 0),
            num_terrain_points + 3: (board_size, board_size, 0),
        }
        terrain_points.update(corner_points)
        # Extract x and y coordinates for Delaunay triangulation
        xy_points = np.array([(point[0], point[1]) for point in terrain_points.values()])
        # Compute Delaunay triangulation
        tri = Delaunay(xy_points)
        # Update cities' z-coordinates
        for i, (x, y, _) in cities.items():
            simplex = tri.find_simplex((x, y))
            if simplex != -1:  # Ensure the point lies within the triangulation
                triangle_indices = tri.simplices[simplex]
                triangle_points = [(*xy_points[idx], terrain_points[idx][2]) for idx in triangle_indices]
                cities[i] = (x, y, plane_z(x, y, triangle_points))
        #Create a combined list of cities and terrain points
        combined_points = {}
        for i, coords in terrain_points.items():
            combined_points[i] = coords
        l = len(combined_points)
        for i, coords in cities.items():
            combined_points[i+l] = coords
        xy_combined_points = np.array([(point[0], point[1]) for point in combined_points.values()])
        tri_2 = Delaunay(xy_combined_points)
        tri_2.simplices = np.concatenate((tri_2.simplices, np.array([[l-4, l-3, l-1], [l-2, l-4, l-1]])), axis = 0)
        faces = [tuple(triangle) for triangle in tri_2.simplices]  # Already a list of tuples
        faces = np.array(faces, dtype=np.int32)  # Use an integer data type for indices
        # Convert vertices to a numpy array
        vertices = list(combined_points.values())  # Already a list of coordinate lists/tuples
        vertices = np.array(vertices, dtype=np.float64)  # Use a float data type for coordinates
        geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        initial_solution = list(cities.keys())  # Example initial solution
        
        start_time = time.time()
        optimal_tour, optimal_distance, optimal_path = simulated_annealing(cities, initial_solution, initial_temperature, cooling_rate)
        end_time = time.time()
        
        runtimes.append(end_time - start_time)
    
    avg_runtime = sum(runtimes) / num_trials
    avg_runtime_sqrt= math.sqrt(avg_runtime)
    runtime_results.append(avg_runtime)
    runtime_results_sqrt.append(avg_runtime_sqrt)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(n_list, runtime_results, marker='o', linestyle='-')
plt.xlabel("Number of Cities")
plt.ylabel("Average Runtime (seconds)")
plt.title("Simulated Annealing Runtime Analysis")
plt.grid()
plt.show()

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(n_list, runtime_results_sqrt, marker='o', linestyle='-')
plt.xlabel("Number of Cities")
plt.ylabel("Square Root of the Average Runtime (seconds)")
plt.title("Simulated Annealing Runtime Analysis")
plt.grid()
plt.show()